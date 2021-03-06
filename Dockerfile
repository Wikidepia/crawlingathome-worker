FROM python:3.8-slim

ARG GIT_COMMIT

RUN apt update && apt install -y rsync git \
    && git clone https://github.com/Wikidepia/crawlingathome-worker --depth=1 \
    && cd crawlingathome-worker \
    && git clone "https://github.com/TheoCoombes/crawlingathome" --depth=1 crawlingathome_client \
    && sed -i 's/np.int64//g' crawlingathome_client/core.py \
    && sed -i 's/import numpy as np//g' crawlingathome_client/core.py \
    && sed -i 's/import numpy as np//g' crawlingathome_client/recycler.py \
    && pip3 install -r requirements.txt \
    && apt clean && rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache \
    && apt remove -y git && apt -y autoremove

WORKDIR crawlingathome-worker

ENV NAME="Anonymous"

ENV GIT_COMMIT=$GIT_COMMIT

CMD python3 crawlingathome.py -n $NAME
