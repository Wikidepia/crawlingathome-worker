import pickle
from multiprocessing import cpu_count

import clip
import tfreecord
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, preprocess):
        self.dataframe = dataframe
        self.image_transform = preprocess
        self.tokenizer = clip.tokenize

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            self.image_transform(Image.open(row["PATH"])),
            self.tokenizer(row["TEXT"], truncate=True)[0],
        )


class CLIP:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        if device == "cpu":
            self.model = torch.quantization.quantize_dynamic(self.model, dtype=torch.qint8)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        with torch.no_grad():
            self.categories = self.model.encode_text(clip.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]).to(device))
            self.underaged_categories = self.model.encode_text(clip.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]).to(device))
            self.animal_categories = self.model.encode_text(clip.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]).to(device))

    def similarity_imgalt(self, image_tensor, text_tokens):
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor.to(device)).float()
            text_features = self.model.encode_text(text_tokens.to(device)).float()
            similarity = self.cosine_similarity(image_features, text_features).tolist()

        image_features = image_features.detach().cpu().numpy()
        return image_features, similarity

    def preprocess_images(self, df):
        ret_image_features = []
        ret_similarity = []
        batch_size = 256 if device == "cuda" else 8
        dataset = CLIPDataset(df, self.preprocess)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=device == "cuda"
        )
        for tensors, tokens in dataloader:
            image_features, similarities = self.similarity_imgalt(tensors, tokens)
            ret_image_features.extend(image_features)
            ret_similarity.extend(similarities)
        return ret_image_features, ret_similarity

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
    tmp_embed = []
    for i, img_embed in enumerate(img_embedding):
        # Round to compensate int8 precision
        if round(similarities[i], 2) < sim_threshold:
            df.drop(i, inplace=True)
            continue

        # Get most similar categories
        nsfw_prob = clip_filter.prob(img_embed, clip_filter.categories)
        df.at[i, "NSFW"] = "UNSURE"
        df.at[i, "similarity"] = similarities[i]
        if nsfw_prob[0] < 19 and nsfw_prob[1] < 19:
            df.at[i, "NSFW"] = "UNLIKELY"
            tmp_embed.append(img_embed)
            continue
        elif nsfw_prob[0] >= 19 and nsfw_prob[1] >= 19:
            df.at[i, "NSFW"] = "NSFW"

        underage_prob = clip_filter.prob(img_embed, clip_filter.underaged_categories)
        if underage_prob[0] < 4 or underage_prob[1] < 4 or any(x in df.at[i, "TEXT"] for x in underaged_text):
            df.drop(i, inplace=True)
            continue

        animal_prob = clip_filter.prob(img_embed, clip_filter.animal_categories)
        if animal_prob[0] > 20:
            df.drop(i, inplace=True)
            continue
        tmp_embed.append(img_embed)
    df.reset_index(drop=True, inplace=True)
    return tmp_embed


def df_tfrecords(df, output_fname):
    writer = tfreecord.RecordWriter()

    def image_to_tfexample(sample_id, image_data, image_format, height, width, caption):
        return {
            "sampleID": writer.bytes_feature(sample_id),
            "image": writer.bytes_feature(image_data),
            "format": writer.bytes_feature(image_format),
            "label": writer.bytes_feature(caption),
            "height": writer.int64_feature(height),
            "width": writer.int64_feature(width),
        }

    with open(output_fname, "ab") as tfr:
        for i in range(len(df)):
            df_image = df.iloc[i]
            image_fname = df_image["PATH"]
            file_type = image_fname.split(".")[-1]
            with open(image_fname, "rb") as f:
                image_data = f.read()
            example = image_to_tfexample(
                str(df_image["SAMPLE_ID"]).encode("utf-8"),
                image_data,
                file_type.encode("utf-8"),
                df_image["HEIGHT"],
                df_image["WIDTH"],
                df_image["TEXT"].encode("utf-8"),
            )
            tfr.write(writer.encode_example(example))


def filter(df, out_fname, output_folder="./save/"):
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
    return len(df)
