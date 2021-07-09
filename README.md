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
