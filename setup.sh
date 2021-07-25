#!/bin/bash

apt-get update
apt-get install -y rsync
if [ ! -f "crawlingathome.py" ]; then
    git clone https://github.com/Wikidepia/crawlingathome-worker
    cd crawlingathome-worker
fi
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir

yes | pip3 uninstall pillow
CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd
