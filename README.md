# Crawling@Home

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

1. `git clone https://github.com/Wikidepia/crawlingathome-worker/`, to download headless-crawlingathome.
2. `cd crawlingathome-worker`, to enter the directory.
3. `python3 -m venv venv && . venv/bin/activate`, to create virtual environment.
4. `. setup.sh`, to install dependencies.
5. `python3 crawlingathome.py`, to start Crawling!

## Droplet Setup
1. use `cloud-config.yaml` script to init the droplet. remember to change to your SSH privatekey in line 9
2. ssh with user `crawl` and check the script by running `tail -f crawl.log`

## TODO
- [x] Save image embedding 
- [x] Convert images to tfrecords
- [x] Upload to google drive
- [x] Prevent corrupt image to be processed
- [x] Shard of chunk (it needs to read all WAT file which will be bad for low ram server)
- [x] Crawling@Home integration
- [ ] Verify output
