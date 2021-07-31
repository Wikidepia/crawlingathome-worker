FROM python:3.8-slim

RUN apt update && apt install -y rsync git \
    && git clone https://github.com/Wikidepia/crawlingathome-worker --depth=1 \
    && cd crawlingathome-worker \
    && pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && git clone "https://github.com/TheoCoombes/crawlingathome" --depth=1 crawlingathome_client \
    && pip3 install -r crawlingathome_client/requirements.txt \
    && pip3 install -r requirements.txt \
    && apt clean && rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache \
    && apt remove -y git && apt -y autoremove

WORKDIR crawlingathome-worker

ENV NAME="Anonymous"

CMD python3 crawlingathome.py -n $NAME
