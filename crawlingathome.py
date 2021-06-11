import gc
import os
import pickle
import shutil
import time
from glob import glob
from urllib.parse import urljoin, urlparse
from uuid import uuid1

import regex
import trio
import ujson
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True  # https://stackoverflow.com/a/47958486


def chunk_using_generators(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def remove_bad_chars(text):
    return regex.sub(r"\p{Cc}|\p{Cs}", "", text)


def parse_wat(content):
    import pycld2 as cld2
    import ftfy

    valid_data = []
    for line in content:
        if "IMG@" not in line:
            continue
        line_str = line.strip()
        data = ujson.loads(line_str)
        linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"][
            "HTML-Metadata"
        ]["Links"]
        base_url = os.path.dirname(
            data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
        )  # get base url

        for e in linklist:
            if "alt" not in e:
                continue
            url = e["url"]
            alt_text = ftfy.fix_text(e["alt"].replace("\n", " ")).strip()
            if url.endswith(".svg") or url.endswith(".gif"):
                continue
            try:
                _, _, details = cld2.detect(alt_text)
            except Exception as e:
                alt_text = remove_bad_chars(alt_text)
                _, _, details = cld2.detect(alt_text)

            if details[0][1] == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)
                valid_data.append((url, alt_text))
    return [
        t for t in {tuple(i) for i in valid_data}
    ]  # Remove duplicate tuple from list


def process_img_content(response, alt_text, sample_id):
    img_output_folder = "save/images/"
    if "content-type" in response.headers:
        if "image/" not in response.headers["content-type"]:
            return
        filetype = (
            response.headers["content-type"].split("/")[-1].split(";")[0]
        )  # Unreliable, maybe get filetype from content?
    else:
        url_path = urlparse(response.url).path
        filetype = os.path.splitext(url_path)[1]

    if "gif" in filetype or "svg" in filetype:
        return

    out_fname = img_output_folder + str(sample_id) + "." + filetype.strip(".")
    try:
        img_data = response.content  # Raise KeyError
        with open(out_fname, "wb") as f:
            f.write(img_data)
        pil_image = Image.open(out_fname)  # Raise UnidentifiedImageError
    except (KeyError, UnidentifiedImageError) as e:
        if os.path.exists(out_fname):
            os.remove(out_fname)
        return
    width, height = pil_image.size
    return [str(sample_id), out_fname, response.url, alt_text, width, height]


async def request_image(datas, start_sampleid):
    import asks

    tmp_data = []
    session = asks.Session(connections=512)

    async def _request(data, sample_id):
        url, alt_text = data
        try:
            proces = process_img_content(
                await session.get(url, timeout=5), alt_text, sample_id
            )
            if proces is not None:
                tmp_data.append(proces)
        except Exception:
            return

    async with trio.open_nursery() as n:
        for data in datas:
            n.start_soon(_request, data, start_sampleid)
            start_sampleid += 1

    with open(f".tmp/{uuid1()}.json", "w") as f:
        ujson.dump(tmp_data, f)
    gc.collect()
    return


async def dl_wat(valid_data, first_sample_id):
    import pandas as pd
    import tractor

    # Download every image available
    processed_samples = []
    async with tractor.open_nursery() as n:
        for i, data in enumerate(chunk_using_generators(valid_data, 4096)):
            await n.run_in_actor(
                request_image, datas=data, start_sampleid=i * 4096 + first_sample_id
            )

    for tmpf in glob(".tmp/*.json"):
        processed_samples.extend(ujson.load(open(tmpf)))
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH"],
    )


def df_clipfilter(df):
    sim_threshold = 0.3
    import clip_filter

    clip = clip_filter.CLIP()
    img_embedding, similarities = clip.preprocess_images(df)
    nsfw_filters = clip.filter(img_embedding, clip.categories)
    underage_filters = clip.filter(img_embedding, clip.underaged_categories)
    # animal_filters = clip.filter(preprocessed_image["img_embedding"], clip.animal_categories)
    for i, (nsfw_prob, underage_prob) in enumerate(zip(nsfw_filters, underage_filters)):

        df.at[i, "similarity"] = similarities[i]
        df.at[i, "NSFW"] = "UNSURE"

        # Review this please, my brain is too smol
        if nsfw_prob <= 19 and underage_prob >= 4:
            df.at[i, "NSFW"] = "UNLIKELY"
        elif nsfw_prob > 19:
            df.at[i, "NSFW"] = "NSFW"

        # Remove image containing underage and not similar image-alttext
        if similarities[i] < sim_threshold or underage_prob < 4:
            df = df.drop(i)
    return df, img_embedding


def df_tfrecords(df, output_folder):
    import tensorflow as tf
    from tfr_image.utils import bytes_feature, int64_feature

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    "sampleID": bytes_feature(sample_id),
                    "image": bytes_feature(image_data),
                    "format": bytes_feature(image_format),
                    "label": bytes_feature(caption),
                    "height": int64_feature(height),
                    "width": int64_feature(width),
                }
            )
        )

    with tf.io.TFRecordWriter(output_folder + "images.tfrecord") as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                df_image["SAMPLE_ID"].encode("utf_8"),
                image_data,
                file_type.encode("utf_8"),
                df_image["HEIGHT"],
                df_image["WIDTH"],
                df_image["TEXT"].encode("utf_8"),
            )
            tfrecord_writer.write(example.SerializeToString())


def upload_gdrive(output_filename):
    import requests

    client_id = (
        "648172777761-onv1nc5f93nhlhf63flsq6onrmjphpfo.apps.googleusercontent.com"
    )
    client_secret = "HZ4Zw-_jVJ-3mwicz1NM5W5x"
    refresh_token = "1//04N2Kysz1LObLCgYIARAAGAQSNwF-L9IrntHNWi2_nEVu2QX5fmlW0Ea0qA-ToBJLSdatDATYxiKcNFI8eZQ_fYN53gjF7b8MGmA"

    def refresh_gdrive_token():
        params = {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
        }

        authorization_url = "https://www.googleapis.com/oauth2/v4/token"

        r = requests.post(authorization_url, data=params)

        if r.ok:
            return r.json()["access_token"]
        else:
            return None

    access_t = refresh_gdrive_token()
    headers = {"Authorization": "Bearer " + access_t}
    para = {
        "name": output_filename.split("/")[-1],
        "parents": ["1CIgcIR7nX2xNBPB577jwEqbbwxAJR_nt"],
    }

    files = {
        "data": ("metadata", ujson.dumps(para), "application/json; charset=UTF-8"),
        "file": ("application/zip", open(output_filename, "rb")),
    }
    requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files,
    )


if __name__ == "__main__":
    import crawlingathome_client as cah

    YOUR_NICKNAME_FOR_THE_LEADERBOARD = "Wikidepia"
    CRAWLINGATHOME_SERVER_URL = "http://178.63.68.247:8181/"

    client = cah.init(
        url=CRAWLINGATHOME_SERVER_URL, nickname=YOUR_NICKNAME_FOR_THE_LEADERBOARD
    )
    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"

    while client.jobCount() > 0:
        start = time.time()
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        if os.path.exists(".tmp"):
            shutil.rmtree(".tmp")

        os.mkdir(output_folder)
        os.mkdir(img_output_folder)
        os.mkdir(".tmp")

        client.newJob()
        client.downloadShard()
        first_sample_id = int(client.start_id)
        last_sample_id = int(client.end_id)

        client.log("Processing shard")
        with open("shard.wat", "r") as infile:
            parsed_data = parse_wat(infile)

        client.log("Downloading images")
        dlparse_df = trio.run(dl_wat, parsed_data, first_sample_id)
        dlparse_df.to_csv(output_folder + "image.csv")

        client.log("Dropping NSFW keywords")
        filtered_df, img_embeddings = df_clipfilter(dlparse_df)
        filtered_df.to_csv(output_folder + "image.csv")
        with open(output_folder + "image_embeddings.pkl", "wb") as f:
            pickle.dump(img_embeddings, f)

        client.log("Saving TFRs")
        df_tfrecords(filtered_df, output_folder)
        # upload_gdrive(output_folder + "image_embeddings.pkl")
        # upload_gdrive(output_folder + "images.tfrecord")
        client._markjobasdone(len(filtered_df))
        print(f"[crawling@home] jobs completed in {round(time.time() - start)} seconds")
