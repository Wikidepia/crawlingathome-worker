import argparse
import hashlib
import logging
import os
import random
import shutil
import subprocess
import tarfile
import time
from io import BytesIO
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import asks
import ftfy
import multiexit
import pandas as pd
import pycld2 as cld2
import requests
import sentry_sdk
import trio
import ujson
from bloom_filter2 import BloomFilter
from PIL import Image, UnidentifiedImageError

import crawlingathome_client as cah


def remove_bad_chars(text):
    return "".join(c for c in text if c.isprintable())


def download_to_file(url, filename):
    headers = {"User-Agent": "Crawling@Home"}
    for _ in range(5):
        try:
            r = requests.get(url, headers=headers)
            with open(filename, "wb") as f:
                f.write(r.content)
            return
        except Exception as e:
            logging.error(f"{e}, sleeping for 5 seconds")
            time.sleep(5)
    raise ValueError(f"Failed to download {url}")


def load_bloom():
    start = time.time()
    for x in ("bloom200M.bin", "clipped.bin", "failed-domains.bin"):
        download_to_file(f"http://the-eye.eu/public/AI/cahblacklists/{x}", f"blocklists/{x}")
    blocklist_dupe = BloomFilter(
        max_elements=200_000_000,
        error_rate=0.05,
        filename=("blocklists/bloom200M.bin", -1),
    )
    blocklist_clipped = BloomFilter(
        max_elements=200_000_000,
        error_rate=0.05,
        filename=("blocklists/clipped.bin", -1),
    )
    blocklist_domain = BloomFilter(
        max_elements=10_000_000,
        error_rate=0.01,
        filename=("blocklists/failed-domains.bin", -1),
    )
    logging.info(f"updated filters in {(time.time()-start):.1f}")
    return blocklist_dupe, blocklist_domain, blocklist_clipped


def parse_wat(fopen):
    valid_data = []
    url_dedupe = set()
    blocklist_dupe, blocklist_domain, blocklist_clipped = load_bloom()
    blocklist_format = set([".svg", ".gif", ".ico", "data:image", "javascript:", "mailto:"])

    for line in fopen:
        if "IMG@" not in line:
            continue
        data = ujson.loads(line)
        links = data["Envelope"]["Payload-Metadata"]["HTTP-Response-Metadata"]["HTML-Metadata"]["Links"]
        base_url = os.path.dirname(data["Envelope"]["WARC-Header-Metadata"]["WARC-Target-URI"])
        img_license = "?"
        for link in links:
            # Check if website is CC License
            if "url" in link and "creativecommons.org/licenses/" in link["url"]:
                img_license = link["url"]
            if "alt" not in link or link["alt"] == "":
                continue
            url = link["url"]
            if not url.startswith("http"):
                url = urljoin(base_url, url)
            alt_text = ftfy.fix_text(link["alt"].replace("\n", " ")).strip()

            try:
                _, _, details = cld2.detect(alt_text)
            except Exception:
                _, _, details = cld2.detect(remove_bad_chars(alt_text))

            if details[0][1] != "en":
                continue

            hashed_imgalt = hashlib.md5((url + alt_text).encode("utf-8")).hexdigest()
            # Skip url with various filter
            try:
                if (
                    any(bl in url.lower() for bl in blocklist_format)
                    or url in url_dedupe
                    or hashed_imgalt in blocklist_dupe
                    or hashed_imgalt in blocklist_clipped
                    or urlparse(url).netloc in blocklist_domain
                ):
                    continue
            except:
                continue

            url_dedupe.add(url)
            valid_data.append((url, alt_text, img_license))
    return valid_data


def process_img_content(response, sample_id):
    img_output_folder = "save/images/"

    try:
        if len(response.content) < 5000:
            return
        img_data = BytesIO(response.content)
        with Image.open(img_data) as im:
            width, height = im.size
            im_format = im.format
            if im_format not in ["JPEG", "PNG", "WEBP"]:
                return
            # Export WEBP image as JPEG
            if im_format == "WEBP":
                im_format = "JPEG"
            if im.mode != "RGB":
                im = im.convert("RGB")
            out_fname = f"{img_output_folder}{sample_id}.{im_format.lower()}"
            im.save(out_fname, im_format)
    except (KeyError, UnidentifiedImageError, Image.DecompressionBombWarning):
        return

    return out_fname, width, height


async def dl_wat(valid_data, first_sample_id):
    cur_sample_id = first_sample_id
    processed_samples = []
    session = asks.Session(connections=192)

    session.headers = {
        "User-Agent": "Crawling at Home Project (http://cah.io.community)",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "http://cah.io.community",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id):
        url, alt_text, license = data
        try:
            process_img = process_img_content(
                await session.get(url, timeout=3, connection_timeout=10, retries=-1),
                sample_id,
            )
            if process_img is not None:
                out_fname, width, height = process_img
                processed_samples.append([str(sample_id), out_fname, url, alt_text, width, height, license])
        except Exception:
            return

    async with trio.open_nursery() as n:
        for data in valid_data:
            cur_sample_id += 1
            n.start_soon(_request, data, cur_sample_id)

    return pd.DataFrame(
        processed_samples,
        columns=["SAMPLE_ID", "PATH", "URL", "TEXT", "HEIGHT", "WIDTH", "LICENSE"],
    )


def upload(source, client_type, target):
    if client_type == "cpu":
        with tarfile.open(f"{source}.tar.gz", "w:gz") as tar:
            tar.add(source, arcname=os.path.basename(source))
        source = f"{source}.tar.gz"

    options = "-av" if client_type == "cpu" else "-zh"
    return os.system(f"rsync {options} {source} {target}")


def chunk_to_shard(fname, shard_piece):
    wc_l = subprocess.run(["wc", "-l", fname], capture_output=True)
    line_count = int(wc_l.stdout.decode("utf-8").split()[0]) // 2

    with open("sharded.wat", "w") as f:
        if shard_piece == 0:
            subprocess.run(["head", "-n", str(line_count), fname], stdout=f)
        elif shard_piece == 1:
            subprocess.run(["tail", "-n", "+" + str(line_count + 1), fname], stdout=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawling@Home Worker")
    parser.add_argument(
        "-n",
        "--nickname",
        type=str,
        required=True,
        help="Nickname for leaderboard",
    )
    parser.add_argument(
        "-t",
        "--type",
        required=False,
        type=str.lower,
        default="hybrid",
        choices=["cpu", "hybrid"],
        help="Worker type selected in the list: cpu, hybrid",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Use debug server & disable upload",
    )
    args = parser.parse_args()
    logging.basicConfig(format="[%(asctime)s crawling@home] %(message)s", datefmt="%H:%M", level=logging.INFO)

    # Setup signal handling to gracefully exit and report on errors
    ignore_errors = [KeyboardInterrupt]
    client_key = "https://dd28610c2d844c0ba0269a2f7cbd088e@o946916.ingest.sentry.io/5897089"
    multiexit.install()
    sentry_sdk.init(client_key, ignore_errors=ignore_errors, release=os.environ["GIT_COMMIT"])
    multiexit.register(lambda: client.bye())

    if args.type == "hybrid":
        import clip_filter
    server_url = "http://cah.io.community/" if not args.debug else "http://178.63.68.247:8181/"
    client = cah.init(url=server_url, nickname=args.nickname, type=args.type)

    output_folder = "./save/"
    img_output_folder = output_folder + "images/"

    while True:
        start = time.time()
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.mkdir(output_folder)
        os.mkdir(img_output_folder)

        try:
            if client.jobCount() < 0:
                break
            client.newJob()
            client.downloadShard()
            first_sample_id = int(client.start_id)
            last_sample_id = int(client.end_id)
            shard_of_chunk = client.shard_piece

            out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
            logging.info(
                f"shard identification {out_fname}"
            )  # in case test fails, we need to remove bad data
            client.log("Processing shard")

            chunk_to_shard("shard.wat", shard_of_chunk)
            with open("sharded.wat", "r") as shard_file:
                parsed_data = parse_wat(shard_file)
            random.shuffle(parsed_data)

            client.log("Downloading images")
            dlparse_df = trio.run(dl_wat, parsed_data, first_sample_id)
            dlparse_df.to_csv(f"{output_folder}{out_fname}.csv", index=False, sep="|")

            if args.type == "hybrid":
                client.log("Dropping NSFW keywords")
                # Filter with local CPU / GPU
                final_images = clip_filter.filter(dlparse_df, out_fname)
                upload_path = f"{output_folder}/*{out_fname}*"
            else:
                # Move result to random uuid folder
                upload_path = uid = uuid4().hex
                final_images = f"rsync {uid}"
                shutil.move("save", uid)

            if not args.debug:
                upload_status = upload(upload_path, args.type, client.upload_address)
                if upload_status != 0:
                    client.log("Upload failed")
                    raise Exception("Upload failed")

            if args.type == "cpu":
                if os.path.exists(f"{upload_path}.tar.gz"):
                    os.remove(f"{upload_path}.tar.gz")
                shutil.rmtree(uid)

            client.completeJob(final_images)
            logging.info(f"jobs completed in {(time.time() - start):.1f} seconds")
        except (cah.core.ServerError, requests.exceptions.ConnectionError):
            logging.error("server error, sleeping for 30 seconds before trying again")
            time.sleep(30)
