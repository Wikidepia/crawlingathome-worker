# Crawling@Home Worker

[![Discord Chat](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/dall-e)

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

### Prebuilt Docker Image

To use worker docker images run the following commands, where $NICKNAME is your nickname that will be showed in leaderboard.

```bash
docker pull wikidepia/crawlingathome-worker
docker run --shm-size 1G -dit -e NAME=$NICKNAME wikidepia/crawlingathome-worker
```

## Contribute

You are more than welcome to contribute to this development, and make it more sane :)
