FROM python:3.8

RUN apt update && apt install rsync -y

RUN curl https://raw.githubusercontent.com/Wikidepia/crawlingathome-worker/master/setup.sh | bash

WORKDIR crawlingathome-worker

ENV NAME="Anonymous"

CMD python3 crawlingathome.py -n $NAME
