# Crawling@Home Worker

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

### Prebuilt Docker Image

To use worker docker images run the following commands, where $NICKNAME is your nickname that will be showed in leaderboard.

```bash
docker pull wikidepia/crawlingathome-worker
docker run -d -e NAME=$NICKNAME wikidepia/crawlingathome-worker
```

You can also manually install crawlingathome-worker, by following this instruction below:

### CPU Worker

1. `git clone https://github.com/Wikidepia/crawlingathome-worker/`, to download crawlingathome-worker.
2. `cd crawlingathome-worker`, to enter the directory.
3. `python3 -m venv venv && . venv/bin/activate`, to create virtual environment, optional for dedicated computers
4. `bash setup.sh`, to install dependencies.
5. `python3 crawlingathome.py -n $NICKNAME`, to start Crawling!

<details><summary>Experimental Feature</summary>

### CPU Worker + TPU (Experimental)

1. Run CPU Worker setup 1-3
2. `bash setup.sh tpu`, to install dependencies.
3. `python3 crawlingathome.py -n $NICKNAME --tpu <ip>:<port>/filter/`, to start Crawling!

## TPU Inference Server Setup (Experimental)

1. You need to setup TPU VM read [here](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm)
2. `pip3 install -r requirements_tpu.txt`, to install required packages.
3. `uvicorn clip_serve-tpu:app --host 0.0.0.0`, to run inference server.

</details>