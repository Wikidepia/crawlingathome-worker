import json
import pickle
import shutil
import time
from glob import glob
from uuid import uuid1

import jax
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from tfr_image.utils import bytes_feature, int64_feature

import clip_jax

DEBUG = True
app = FastAPI()

image_fn, text_fn, jax_params, jax_preprocess = clip_jax.load("ViT-B/32", "cpu")
devices = jax.local_devices()

categories = text_fn(jax_params, clip_jax.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]),)
underaged_categories = text_fn(jax_params, clip_jax.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]))
animal_categories = text_fn(jax_params, clip_jax.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]))

jax_params = jax.device_put_replicated(jax_params, devices)
image_fn = jax.pmap(image_fn)
text_fn = jax.pmap(text_fn)


def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# I have no idea about this jax.jit stuff (1)
@jax.jit
def prob(image_features, text_embed):
    image_features = jax.numpy.expand_dims(image_features, 0)
    image_features /= jax.numpy.linalg.norm(image_features, axis=-1, keepdims=True)
    image_sim = jax.nn.softmax(100.0 * image_features @ text_embed.transpose())
    return jax.numpy.argsort(-image_sim)[0]

# I have no idea about this jax.jit stuff (2)
@jax.jit
def cosine_similarity(image_features, text_features):
    image_features = jax.numpy.expand_dims(image_features, 0)

    image_features_norm = jax.numpy.linalg.norm(image_features, axis=1, keepdims=True)
    text_features_norm = jax.numpy.linalg.norm(text_features, axis=0, keepdims=True)

    # Distance matrix of size (b, n).
    return (
        (image_features @ text_features) / (image_features_norm @ text_features_norm)
    ).T


def process_text_batch(texts, bs, n_device):
    jax_texts = []
    if len(texts) < bs * n_device:
        rem = bs * n_device - len(texts)
        texts.extend(["a photo of dog"] * rem)
    for tx in split_list(texts, bs):
        jax_texts.append(clip_jax.tokenize(tx))
    return jax.numpy.asarray(jax_texts)


def process_image_batch(file_list, bs, n_device):
    jax_images = []
    if len(file_list) < bs * n_device:
        rem = bs * n_device - len(file_list)
        file_list.extend([None] * rem)

    empty_img = Image.new("RGB", (244, 244))
    for batch in split_list(file_list, bs):
        images = []
        for im in batch:
            im = empty_img if im == None else Image.open(im)
            images.append(jax_preprocess(im))
        jax_images.append(images)
    return jax.numpy.asarray(jax_images)


def clip_filter(embeddings, similarity, df):
    sim_threshold = 0.3
    ret_embeddings = []
    underaged_text = ["teen", "kid", "child", "baby"]
    for i, (im, similarity) in enumerate(zip(embeddings, similarity)):
        if round(similarity, 2) < sim_threshold:
            df.drop(i, inplace=True)
            continue

        # get most similar categories
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = float(similarity)
        nsfw_prob = prob(im, categories)
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            ret_embeddings.append(im)
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        underage_prob = prob(im, underaged_categories)
        if (
            underage_prob[0] < 4
            or underage_prob[1] < 4
            or any(x in df.at[i, "TEXT"] for x in underaged_text)
        ):
            df.drop(i, inplace=True)
            continue

        animal_prob = prob(im, animal_categories)
        if animal_prob[0] > 20:
            df.drop(i, inplace=True)
            continue

        ret_embeddings.append(im)
    df.reset_index(drop=True, inplace=True)
    return ret_embeddings


def generate_embeddings(images, texts, batch_size):
    global devices
    img_result = []
    sim_result = []
    img_text_files = [[im, tx] for im, tx in zip(images, texts)]
    batches = split_list(img_text_files, batch_size * len(devices))
    for batch in batches:
        result = []
        processed_img = process_image_batch(
            [x[0] for x in batch], batch_size, len(devices)
        )
        processed_text = process_text_batch(
            [x[1][:75] for x in batch], batch_size, len(devices)
        )
        jax_image_embed = image_fn(jax_params, processed_img)
        jax_text_embed = text_fn(jax_params, processed_text)

        for ims, txs in zip(jax_image_embed, jax_text_embed):
            for im, tx in zip(ims, txs):
                result.append((im, cosine_similarity(im, tx)[0]))
        real_result = result[: len(batch)]
        img_result.extend([x[0] for x in real_result])
        sim_result.extend([x[1] for x in real_result])
    return img_result, sim_result


def df_tfrecords(df, output_fname, uuid):
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

    with tf.io.TFRecordWriter(output_fname) as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(uuid + "/" + image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image["SAMPLE_ID"]).encode("utf_8"),
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
        "data": ("metadata", json.dumps(para), "application/json; charset=UTF-8"),
        "file": ("application/zip", open(output_filename, "rb")),
    }
    requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files,
    )


@app.post("/filter/")
async def cah_clip(file: UploadFile = File(...)):
    start = time.time()
    uuid = str(uuid1())
    batch_size = 128
    print(f"[{uuid}] Received jobs")
    with open(f"{uuid}.zip", "wb") as f:
        f.write(await file.read())
    shutil.unpack_archive(f"{uuid}.zip", uuid)

    csv_file = glob(f"{uuid}/save/*.csv")[0]
    out_fname = csv_file.split("/")[-1].replace(".csv", "")
    df = pd.read_csv(csv_file, sep="|")
    images = df["PATH"].tolist()
    texts = df["TEXT"].tolist()

    images = [uuid + "/" + x for x in images]
    print(f"[{uuid}] Generating embeddings")
    embeddings, similarity = generate_embeddings(images, texts, batch_size)
    print(f"[{uuid}] Filter with CLIP")
    img_embeddings = clip_filter(embeddings, similarity, df)

    print(f"[{uuid}] Save embeddings to pickle")
    img_embeds_sampleid = {}
    for i, img_embed_it in enumerate(img_embeddings):
        dfid_index = df.at[i, "SAMPLE_ID"]
        img_embeds_sampleid[str(dfid_index)] = img_embed_it
    with open(f"{uuid}/save/image_embedding_dict-{out_fname}.pkl", "wb") as f:
        pickle.dump(img_embeds_sampleid, f)

    df.to_csv(csv_file, index=False, sep="|")
    print(f"[{uuid}] Save image to tfrecords")
    df_tfrecords(
        df, f"{uuid}/save/crawling_at_home_{out_fname}__00000-of-00001.tfrecord", uuid
    )
    if not DEBUG:
        upload_gdrive(f"{uuid}/save/image_embedding_dict-{out_fname}.pkl")
        upload_gdrive(
            f"{uuid}/save/crawling_at_home_{out_fname}__00000-of-00001.tfrecord"
        )
        upload_gdrive(csv_file)
    print(f"[{uuid}] Jobs completed in {time.time() - start}")
    return {"len_result": len(df)}