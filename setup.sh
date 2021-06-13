#!/bin/bash

sudo apt-get update
sudo apt-get install -y git build-essential python3-dev python3-pip python3-venv
if [ ! -f "crawlingathome.py" ]; then
    git clone https://github.com/Wikidepia/headless-crawlingathome
    cd headless-crawlingathome
    python3 -m venv venv && . venv/bin/activate
fi
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir
pip3 install git+https://github.com/Wikidepia/CLIP --no-cache-dir
