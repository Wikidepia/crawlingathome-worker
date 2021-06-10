#!/bin/bash

git clone "https://github.com/Wikidepia/CLIP" CLOP
mv CLOP/clip .
rm -rf CLOP
git clone "https://github.com/TheoCoombes/crawlingathome" crawlingathome_client
pip3 install -r crawlingathome_client/requirements.txt
pip3 install -r requirements.txt
