# Crawling@Home Worker

# Deprecated! Please use [ARKseal/crawlingathome-worker](https://github.com/ARKseal/crawlingathome-worker)

[![Discord Chat](https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white)](https://discord.gg/dall-e) [![Docker Image](https://img.shields.io/docker/image-size/wikidepia/crawlingathome-worker)](https://hub.docker.com/repository/docker/wikidepia/crawlingathome-worker)


> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

### Prebuilt Docker Image

To use worker docker images run the following commands, where $NICKNAME is your nickname that will be showed in leaderboard.

```bash
docker run --detach \
  --name watchtower \
  --restart=on-failure \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower --label-enable --cleanup --interval 1800 && \
docker run --detach -it \
  --name crawlingathome-worker \
  --label=com.centurylinklabs.watchtower.enable=true \
  --restart=on-failure \
  --shm-size 1G \
  -e NAME=$NICKNAME \
  wikidepia/crawlingathome-worker:latest-cpu
```

### Build Docker Image

You can build docker image yourself with the following command

```bash
docker build --no-cache --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
```

## Contribute

You are more than welcome to contribute to this development, and make it more sane :)
