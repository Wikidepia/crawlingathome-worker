FROM python:3.8-slim

ARG GIT_COMMIT

RUN apt update && apt install -y rsync git \
    && git clone https://github.com/Wikidepia/crawlingathome-worker --depth=1 \
    && cd crawlingathome-worker \
    && pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 install git+https://github.com/openai/CLIP \
    && pip3 install pandas tfreecord \
    && git clone "https://github.com/TheoCoombes/crawlingathome" --depth=1 crawlingathome_client \
    && pip3 install -r crawlingathome_client/requirements.txt \
    && pip3 install -r requirements.txt \
    && apt clean && rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache \
    && apt remove -y git && apt -y autoremove

WORKDIR crawlingathome-worker

ENV NAME="Anonymous"

ENV GIT_COMMIT=$GIT_COMMIT

CMD python3 crawlingathome.py -n $NAME
