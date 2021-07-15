# Crawling@Home Worker

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

### Prebuilt Docker Image

To use worker docker images run the following commands, where $NICKNAME is your nickname that will be showed in leaderboard.

```bash
docker pull wikidepia/crawlingathome-worker
docker run --shm-size 4G -dit -e NAME=$NICKNAME wikidepia/crawlingathome-worker
```
