import os
import pickle
import re
from urllib.parse import urljoin, urlparse

import asks
import pandas as pd
import pycld2 as cld2
import tensorflow as tf
import trio
import ujson
from PIL import Image
from tfr_image.utils import bytes_feature, int64_feature

import clip_filter

clip = clip_filter.CLIP()


def remove_bad_chars(text):
    return re.sub(r"\p{Cc}|\p{Cs}", "", text)


def parse_wat(content):
    urlist = []
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
            alt_text = e["alt"].encode("ascii", "ignore").decode()
            for _ in range(2):
                try:
                    _, _, details = cld2.detect(alt_text)
                    break
                except Exception as e:
                    alt_text = remove_bad_chars(alt_text)

            if details[0][1] == "en":
                if not url.startswith("http"):
                    url = urljoin(base_url, url)
                valid_data.append((url, alt_text))
    return [
        t for t in (set(tuple(i) for i in valid_data))
    ]  # Remove duplicate tuple from list


async def request_image(data):
    global responses
    url, alt_text = data
    try:
        r = await asks.get(url, timeout=120)
        if len(r.content) < 1024:
            return
    except Exception:
        return
    return responses.append((r, alt_text))


async def dl_wat(valid_data, first_sample_id, img_output_folder):
    global responses
    responses = []
    processed_samples = []
    sample_id = first_sample_id

    # Download every image available
    async with trio.open_nursery() as nursery:
        for data in valid_data:
            nursery.start_soon(request_image, data)
    print("Download part is done")

    for (response, alt_text) in responses:
        # filetype from mime
        if "content-type" in response.headers:
            if "image/" not in response.headers["content-type"]:
                continue
            filetype = (
                response.headers["content-type"].split("/")[-1].split(";")[0]
            )  # Unreliable, maybe get filetype from content?
        else:
            url_path = urlparse(response.url).path
            filetype = os.path.splitext(url_path)[1]

        if "gif" in filetype or "svg" in filetype:
            continue

        img_data = response.content
        out_fname = img_output_folder + str(sample_id) + "." + filetype.strip(".")
        with open(out_fname, "wb") as f:
            f.write(img_data)

        pil_image = Image.open(out_fname)
        width, height = pil_image.size
        processed_samples.append(
            [str(sample_id), out_fname, response.url, alt_text, width, height]
        )
        sample_id += 1
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH"],
    )


def df_clipfilter(df):
    nsfw_filters, img_embeddings = clip.filter(df, clip.categories)
    underage_filters, _ = clip.filter(df, clip.underaged_categories)
    animal_filters, _ = clip.filter(df, clip.animal_categories)
    for i, (nsfw_prob, underage_prob, animal_prob) in enumerate(
        zip(nsfw_filters, underage_filters, animal_filters)
    ):
        nsfw_prob, underage_prob = nsfw_prob["probs"], underage_prob["probs"]

        # Review this please, my brain is too smol
        if nsfw_prob <= 19 and underage_prob >= 4:
            df.at[i, "NSFW"] = "UNLIKELY"
        elif nsfw_prob > 19 or underage_prob < 4:
            df.at[i, "NSFW"] = "NSFW"
        else:
            df.at[i, "NSFW"] = "UNSURE"
    return df, img_embeddings


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


def df_tfrecords(df, output_folder):
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


if __name__ == "__main__":
    output_folder = "./save/"
    csv_output_folder = output_folder
    img_output_folder = output_folder + "images/"
    similarity_threshold = 0.3
    first_sample_id = 0

    with open("shard.wat", "r") as infile:
        parsed_data = parse_wat(infile)
    dlparse_df = trio.run(
        dl_wat, parsed_data[:3000], first_sample_id, img_output_folder
    )
    filtered_df, img_embeddings = df_clipfilter(dlparse_df)
    with open(output_folder + "image_embeddings.pkl", "wb") as f:
        pickle.dump(img_embeddings, f)
    df_tfrecords(filtered_df, output_folder)
