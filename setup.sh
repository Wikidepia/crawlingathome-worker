#!/bin/bash

sudo apt-get update
sudo apt-get install -y git build-essential python3-dev python3-pip python3-venv libjpeg-dev
if [ ! -f "crawlingathome.py" ]; then
    git clone https://github.com/Wikidepia/crawlingathome-worker
    cd crawlingathome-worker
    python3 -m venv venv && . venv/bin/activate
fi
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt --no-cache-dir
pip3 install -r requirements.txt --no-cache-dir
pip3 install tensorflow --no-cache-dir
pip3 install git+https://github.com/Wikidepia/CLIP --no-cache-dir

ssh-keygen -t rsa -b 4096 -f $HOME/.ssh/id_cah -q -P ""
sed -i -e "s/<<your_ssh_public_key>>/$(sed 's:/:\\/:g' ~/.ssh/id_cah.pub)/" cloud-config.yaml

yes | pip3 uninstall pillow
CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd
