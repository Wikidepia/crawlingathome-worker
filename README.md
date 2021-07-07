# Crawling@Home Worker

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

0. You need to have TPU inference server, and Linux OS.
1. `git clone https://github.com/Wikidepia/crawlingathome-worker/`, to download headless-crawlingathome.
2. `cd crawlingathome-worker`, to enter the directory.
3. `python3 -m venv venv && . venv/bin/activate`, to create virtual environment, optional for dedicated computers
4. `bash setup.sh`, to install dependencies.
5. `python3 crawlingathome.py -n <username> --tpu <api_url>`, to start Crawling!

## TPU Inference Setup

1. You need to setup TPU VM read [here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)
2. `pip3 install -r requirements_tpu.txt`, to install required packages.
3. `uvicorn clip_serve-tpu:app --host 0.0.0.0`, to run inference server.
