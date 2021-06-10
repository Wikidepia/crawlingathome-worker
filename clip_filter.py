import numpy as np
from datasets import Dataset
from PIL import Image


class CLIP:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.categories = ["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]
        self.underaged_categories = ["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]
        self.animal_categories = ["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]

    def load_img(self, path):
        image = Image.open(path)
        rgbimg = Image.new("RGB", image.size)
        rgbimg.paste(image)
        return rgbimg

    def classification(self, batch):
        inputs = self.processor(
            images=[self.load_img(f) for f in batch["PATH"]],
            text=batch["classes"][0].split("**"),
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = (
            outputs.logits_per_image
        )  # this is the image-text similarity score
        batch["probs"] = logits_per_image.softmax(dim=1).detach().numpy()
        batch["img_embeddings"] = outputs.image_embeds.detach().numpy()
        return batch

    def filter(self, df, classes):
        ret = []
        df["classes"] = "**".join(classes)
        dataset = Dataset.from_pandas(df)
        result = dataset.map(self.classification, batched=True, batch_size=8)
        for path, probs in zip(result["PATH"], result["probs"]):
            max_probs = np.argmax(probs)
            ret.append({"path": path, "probs": max_probs})
        del df["classes"]
        return ret, result["img_embeddings"]
