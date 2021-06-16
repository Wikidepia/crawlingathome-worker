# Crawling@Home

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

1. `git clone https://github.com/Wikidepia/crawlingathome-worker/`, to download headless-crawlingathome.
2. `cd crawlingathome-worker`, to enter the directory.
3. `python3 -m venv venv && . venv/bin/activate`, to create virtual environment, optional for dedicated computers
4. `bash setup.sh`, to install dependencies.
5. `python3 crawlingathome.py`, to start Crawling!

## Droplet Setup
1. use `cloud-config.yaml` script to init the droplet
2. ssh with this command `ssh -oIdentitiesOnly=yes -i~/.ssh/id_cah crawl@{your-droplet-ip}}`
3. check the script by running `tail -f crawl.log`

## TODO
- [x] Save image embedding 
- [x] Convert images to tfrecords
- [x] Upload to google drive
- [x] Prevent corrupt image to be processed
- [x] Shard of chunk (it needs to read all WAT file which will be bad for low ram server)
- [x] Crawling@Home integration
- [x] Verify output
