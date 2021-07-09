#!/bin/bash

sudo apt-get update
sudo apt-get install -y git build-essential python3-dev python3-pip libjpeg-dev tmux
if [ ! -f "crawlingathome.py" ]; then
    git clone https://github.com/Wikidepia/crawlingathome-worker
    cd crawlingathome-worker
fi
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir
pip3 install asks ftfy --no-cache-dir

pip3 install tensorflow tfr_image datasets --no-cache-dir
pip3 install git+https://github.com/Wikidepia/CLIP --no-cache-dir
yes | pip3 uninstall pillow
CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd
