# Crawling@Home

> Help us build a billion-scale image-caption dataset by filtering Common Crawl with OpenAI CLIP

## Setup

1. `git clone https://github.com/Wikidepia/headless-crawlingathome/`, to download headless-crawlingathome.
2. `cd headless-crawlingathome`, to enter the directory.
3. `python3 -m venv venv && . venv/bin/activate`, to create virtual environment.
4. `. setup.sh`, to install dependencies.
5. `python3 crawlingathome.py`, to start Crawling!

## TODO
- [x] Save image embedding 
- [x] Convert images to tfrecords
- [x] Upload to google drive
- [x] Prevent corrupt image to be processed
- [ ] Shard of chunk (it needs to read all WAT file which will be bad for low ram server)
- [x] Crawling@Home integration
- [ ] Verify output
