import numpy as np
import torch
from datasets import Dataset
from PIL import Image

import clip

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIP:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        if device == "cpu":
            self.model = torch.quantization.quantize_dynamic(self.model, dtype=torch.qint8)
        self.tokenize = clip.tokenize
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_img(self, path):
        image = Image.open(path)
        rgbimg = Image.new("RGB", image.size)
        rgbimg.paste(image)
        return rgbimg

    def _preprocess_images(self, batch):
        similarity = []
        images = [
            self.preprocess(self.load_img(path)).unsqueeze(0).to(device)
            for path in batch["PATH"]
        ]
        texts = [self.tokenize(text[:77]).to(device) for text in batch["TEXT"]]

        with torch.no_grad():
            image_features = self.model.encode_image(
                torch.cat([x.float() for x in images])
            )
            text_features = self.model.encode_text(torch.cat([x for x in texts]))

        for image_feat, text_feat in zip(image_features, text_features):
            similarity.append(
                float(
                    self.cosine_similarity(
                        torch.reshape(text_feat, (1, 512)),
                        torch.reshape(image_feat, (1, 512)),
                    )
                )
            )

        batch["similarity"] = similarity
        batch["image_features"] = image_features.detach().cpu().numpy()
        return batch

    def preprocess_images(self, df):
        im_dataset = Dataset.from_pandas(df)
        im_dataset = im_dataset.map(self._preprocess_images, batched=True, batch_size=8)
        return im_dataset["image_features"], im_dataset["similarity"]

    def filter(self, img_embeddings, classes):
        ret = []
        text = self.model.encode_text(self.tokenize(classes).to(device))

        with torch.no_grad():
            for emb in img_embeddings:
                logits_per_image, _ = self.model(
                    torch.as_tensor(emb).to(device), text.float()
                )
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                ret.append(np.argmax(probs))
        return ret
