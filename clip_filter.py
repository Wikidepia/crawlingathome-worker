import clip
from copy import copy
import datasets
import torch
from PIL import Image
import requests
import json
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
datasets.set_caching_enabled(False)

_tokenizer = clip.simple_tokenizer.SimpleTokenizer()


def clip_tokenize(texts, context_length=77):
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            tokens = tokens[:76] + [eot_token]
        result[i, : len(tokens)] = torch.tensor(tokens)
    return result


class CLIP:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        with torch.no_grad():
            self.categories = self.model.encode_text(clip_tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]).to(device))
            self.underaged_categories = self.model.encode_text(clip_tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]).to(device))
            self.animal_categories = self.model.encode_text(clip_tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]).to(device))

    def similarity_imgalt(self, batch):
        similarity = []
        images = [
            self.preprocess(Image.open(path)).unsqueeze(0).to(device)
            for path in batch["PATH"]
        ]
        texts = clip_tokenize(batch["TEXT"]).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(torch.cat(images)).float()
            text_features = self.model.encode_text(texts).float()
            similarity = self.cosine_similarity(image_features, text_features).tolist()

        batch["similarity"] = similarity
        batch["image_features"] = image_features.detach().cpu().numpy()
        return batch

    def preprocess_images(self, df):
        im_dataset = datasets.Dataset.from_pandas(df)
        im_dataset = im_dataset.map(self.similarity_imgalt, batched=True, batch_size=8)
        return im_dataset["image_features"], im_dataset["similarity"]

    def prob(self, image_features, text_features):
        text_features = text_features.float()
        image_features = torch.as_tensor(image_features).to(device, dtype=torch.float32)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity.topk(2)
        return indices


clip_filter = CLIP()


def df_clipfilter(df):
    sim_threshold = 0.3
    underaged_text = ["teen", "kid", "child", "baby"]

    img_embedding, similarities = clip_filter.preprocess_images(df)
    tmp_embed = copy(img_embedding)
    for i, img_embed in enumerate(tmp_embed):
        if similarities[i] < sim_threshold:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        # get most similar categories
        nsfw_prob = clip_filter.prob(img_embed, clip_filter.categories)
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarities[i]
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        underage_prob = clip_filter.prob(img_embed, clip_filter.underaged_categories)
        if (
            underage_prob[0] < 4
            or underage_prob[1] < 4
            or any(x in df.at[i, "TEXT"] for x in underaged_text)
        ):
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)
            continue

        animal_prob = clip_filter.prob(img_embed, clip_filter.animal_categories)
        if animal_prob[0] > 20:
            df.drop(i, inplace=True)
            img_embedding.remove(img_embed)

    df.reset_index(drop=True, inplace=True)
    return img_embedding


def df_tfrecords(df, output_fname):
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

    with tf.io.TFRecordWriter(output_fname) as tfrecord_writer:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with tf.io.gfile.GFile(image_fname, "rb") as f:
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


def filter(df, out_fname, output_folder="./save/", debug=False):
    img_embeddings = df_clipfilter(df)
    df.to_csv(f"{output_folder}{out_fname}.csv", index=False, sep="|")

    img_embeds_sampleid = {}
    for i, img_embed_it in enumerate(img_embeddings):
        dfid_index = df.at[i, "SAMPLE_ID"]
        img_embeds_sampleid[str(dfid_index)] = img_embed_it
    with open(f"{output_folder}image_embedding_dict-{out_fname}.pkl", "wb") as f:
        pickle.dump(img_embeds_sampleid, f)

    print(f"[crawling@home] filtered pairs: {len(df)}")
    df_tfrecords(
        df,
        f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord",
    )
    if not debug:
        upload_gdrive(f"{output_folder}image_embedding_dict-{out_fname}.pkl")
        upload_gdrive(
            f"{output_folder}crawling_at_home_{out_fname}__00000-of-00001.tfrecord"
        )
        upload_gdrive(output_folder + out_fname + ".csv")
    return len(df)
