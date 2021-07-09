FROM python:3.8

RUN curl https://raw.githubusercontent.com/Wikidepia/crawlingathome-worker/master/setup.sh | bash

WORKDIR crawlingathome-worker

ENV NAME="Anonymous"

CMD git pull

CMD python3 crawlingathome.py -n $NAME
