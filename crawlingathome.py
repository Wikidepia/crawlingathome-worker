import json
import os
import time
from urllib.parse import urljoin, urlparse

import asks
import cairosvg
import pandas as pd
import pycld2 as cld2
import trio

import clip_filter

clip = clip_filter.CLIP()
output_folder = "./save/"
csv_output_folder = output_folder
img_output_folder = output_folder + "images/"
similarity_threshold = 0.3
start_time = time.time()
first_sample_id = 0


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


def parse_wat(content):
    urlist = []
    valid_data = []
    for line in content:
        line_str = line.strip()
        if "IMG@" in line_str and "alt:" in line_str:
            data = json.loads(line_str)
            linklist = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"][
                "HTML-Metadata"
            ]["Links"]
            base_url = os.path.dirname(
                data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"]
            )  # get base url

            for e in linklist:
                if "alt" not in e or len(e["alt"]) < 4:
                    continue
                url = e["url"]
                alt_text = e["alt"].encode("ascii", "ignore").decode()
                _, _, details = cld2.detect(alt_text[:76])
                if details[0][1] == "en" and url not in urlist:
                    if not url.startswith("http"):
                        url = urljoin(base_url, url)
                    urlist.append(url)
                    valid_data.append((url, alt_text))
    return valid_data


async def dl_wat(
    valid_data,
    first_sample_id,
):
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

        img_data = response.content
        if "svg" in filetype:  # Untested
            filetype = "png"
            img_data = cairosvg.svg2png(
                url=response.url,
                output_height=600,
            )
        elif "gif" in filetype:
            continue
        out_fname = img_output_folder + str(sample_id) + "." + filetype.strip(".")
        with open(out_fname, "wb") as f:
            f.write(img_data)

        processed_samples.append([str(sample_id), out_fname, response.url, alt_text])
        sample_id += 1
    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT"],
    )


def df_clipfilter(df):
    categories = ["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]
    underaged_categories = ["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]
    animal_categories = ["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]
  
    nsfw_filters = clip.clip_filter(df, categories)
    underage_filters = clip.clip_filter(df, underaged_categories)
    animal_filters = clip.clip_filter(df, animal_categories)
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
        # print(categories[nsfw_prob], underaged_categories[underage_prob])
    return df


if True:
    with open("shard.wat", "r") as infile:
        parsed_data = parse_wat(infile)
    dlparse_df = trio.run(dl_wat, parsed_data, first_sample_id)
