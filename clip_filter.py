import numpy as np
from datasets import Dataset
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIP:
    def __init__(self):
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def load_img(self, path):
        image = Image.open(path)
        if ".png" in path:
            image = image.convert("RGB")
        return image

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
        return batch

    def filter(self, df, classes):
        ret = []
        df["classes"] = "**".join(classes)
        dataset = Dataset.from_pandas(df)
        result = dataset.map(self.classification, batched=True, batch_size=8)
        for i, x in enumerate(result["PATH"]):
            probs = result["probs"][i]
            max_probs = np.argmax(probs)
            ret.append({"path": x, "probs": max_probs})
        del df["classes"]
        del df["PATH"]
        return ret
