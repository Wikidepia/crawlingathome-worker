import argparse
import csv
import hashlib
import logging
import os
import random
import shutil
import ssl
import subprocess
import tarfile
import time
from io import BytesIO, StringIO
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import asks
import ftfy
import multiexit
import pycld2 as cld2
import requests
import sentry_sdk
import trio
import ujson
from bloom_filter2 import BloomFilter
from PIL import Image, UnidentifiedImageError

import crawlingathome_client as cah
from crawlingathome_client.temp import TempCPUWorker

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE


def remove_bad_chars(text):
    return "".join(c for c in text if c.isprintable())


def load_bloom():
    url = "http://the-eye.eu/public/AI/cahblacklists/failed-domains.bin"
    if not os.path.exists("failed-domains.bin"):
        r = requests.get(url, headers={"User-Agent": "Crawling@Home"})
        with open("failed-domains.bin", "wb") as f:
            f.write(r.content)

    blocklist_domain = BloomFilter(
        max_elements=10_000_000,
        error_rate=0.01,
        filename=("failed-domains.bin", -1),
    )
    url_dedupe = BloomFilter(
        max_elements=100_000_000,
        error_rate=0.01,
        filename=("url-filter.bin", -1),
    )
    return blocklist_domain, url_dedupe


def parse_wat(fopen):
    valid_data = []
    blocklist_domain, url_dedupe = load_bloom()
    wat_url = set()
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
            alt_text = ftfy.fix_text(link["alt"].replace("\n", " ")).strip()

            try:
                _, _, details = cld2.detect(alt_text)
            except Exception:
                _, _, details = cld2.detect(remove_bad_chars(alt_text))

            if details[0][1] != "en":
                continue

            if not url.startswith("http"):
                url = urljoin(base_url, url)
            hashed_imgalt = hashlib.md5((url + alt_text).encode("utf-8")).hexdigest()
            # Skip url with various filter
            try:
                if not (
                    any(bl in url.lower() for bl in blocklist_format)
                    or url in url_dedupe
                    or url in wat_url
                    or urlparse(url).netloc in blocklist_domain
                    or len(url) > 2048  # prevent bufio.scanner too long
                ):
                    valid_data.append((url, alt_text, img_license, hashed_imgalt))
                    wat_url.add(url)
            except:
                pass

    # Deduplicate to bloom filter server
    hashes = StringIO("\n".join(x[3] for x in valid_data))
    req_bloom = requests.post(
        "http://116.202.162.146:8000/deduplicate/", files={"file": hashes}, data={"key": "clipped"}
    )
    deduped_hashes = set(req_bloom.text.split("\n"))
    valid_data = [x for x in valid_data if x[3] in deduped_hashes]
    map(url_dedupe.add, (x[0] for x in valid_data)) # Add to URL bloom filter
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
            exif = im.info.get("exif", b"")
            out_fname = f"{img_output_folder}{sample_id}.{im_format.lower()}"
            if im_format not in ["JPEG", "PNG", "WEBP"]:
                return
            if im.mode != "RGB":
                im = im.convert("RGB")
                im.save(out_fname, im_format, exif=exif)
            else:
                with open(out_fname, "wb") as f:
                    f.write(response.content)
    except (KeyError, UnidentifiedImageError, Image.DecompressionBombWarning):
        return

    return out_fname, width, height


async def dl_wat(valid_data, cur_sample_id):
    processed_samples = []
    session = asks.Session(connections=192, ssl_context=ssl_ctx)

    session.headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
        "Accept-Language": "en-US",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://google.com",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async def _request(data, sample_id):
        url, alt_text, license, _ = data
        try:
            process_img = process_img_content(
                await session.get(url, timeout=10, connection_timeout=20, retries=-1),
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

    return processed_samples


def upload(source, target):
    with tarfile.open(f"{source}.tar.gz", "w:gz") as tar:
        tar.add(source, arcname=os.path.basename(source))
    source = f"{source}.tar.gz"

    return os.system(f"rsync -a --remove-source-files {source} {target}")


def chunk_to_shard(fname):
    wc_l = subprocess.run(["wc", "-l", fname], capture_output=True)
    line_count = int(wc_l.stdout.decode("utf-8").split()[0]) // 2

    for shard_piece in range(2):
        with open(f"sharded-{shard_piece}.wat", "w") as f:
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

    server_url = "http://cah.io.community/" if not args.debug else "http://178.63.68.247:8181/"
    client = TempCPUWorker(url=server_url, nickname=args.nickname)

    output_folder = "./save/"
    img_output_folder = output_folder + "images/"
    url_dedupe_count = 0

    if os.path.exists("url-filter.bin"):
        os.remove("url-filter.bin")

    while True:
        start = time.time()
        try:
            if client.jobCount() < 0:
                break
            complete = {}
            client.newJob()
            client.downloadWat()

            chunk_to_shard("shard.wat")
            # 90M to prevent overcapacity
            if url_dedupe_count > 90_000_000:
                os.remove("url-filter.bin")
            for shard_of_chunk in range(2):
                if os.path.exists(output_folder):
                    shutil.rmtree(output_folder)

                os.mkdir(output_folder)
                os.mkdir(img_output_folder)

                first_sample_id = int(client.shards[shard_of_chunk][1]["start_id"])
                last_sample_id = int(client.shards[shard_of_chunk][1]["end_id"])

                out_fname = f"FIRST_SAMPLE_ID_IN_SHARD_{str(first_sample_id)}_LAST_SAMPLE_ID_IN_SHARD_{str(last_sample_id)}_{shard_of_chunk}"
                logging.info(f"Shard ID : {out_fname}")
                logging.info("Processing shard")
                client.log("Processing shard")

                with open(f"sharded-{shard_of_chunk}.wat", "r") as shard_file:
                    parsed_data = parse_wat(shard_file)
                url_dedupe_count += len(parsed_data)
                random.shuffle(parsed_data)

                logging.info("Downloading images")
                dlparse_df = trio.run(dl_wat, parsed_data, first_sample_id)
                logging.info(f"Successfully download {len(dlparse_df)} images out of {len(parsed_data)} links")

                with open(f"{output_folder}{out_fname}.csv", "w") as outfile:
                    writer = csv.writer(outfile, delimiter="|")
                    writer.writerow(["SAMPLE_ID", "PATH", "URL", "HEIGHT", "WIDTH", "LICENSE"])
                    writer.writerows(dlparse_df)

                # Move result to random uuid folder
                upload_path = uuid4().hex
                shutil.move("save", upload_path)

                if not args.debug:
                    upload_status = upload(upload_path, client.upload_address)
                    if upload_status != 0:
                        client.log("Upload failed")
                        raise Exception("Upload failed")

                shutil.rmtree(upload_path)
                complete[str(client.shards[shard_of_chunk][0])] = f"rsync {upload_path}"
            client.completeJob(complete)
            logging.info(f"Jobs completed in {(time.time() - start):.1f} seconds")
        except (cah.core.ServerError, requests.exceptions.ConnectionError):
            logging.error("Server error, sleeping for 30 seconds before trying again")
            time.sleep(30)
