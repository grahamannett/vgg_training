FROM python:3.7.8

ENV DATASET_NAME "imagenette2-160"
ENV DATASET_URL "https://s3.amazonaws.com/fast-ai-imageclas/${DATASET_NAME}.tgz"

RUN wget --directory-prefix=/root/data $DATASET_URL

COPY requirements.txt ./
RUN pip install -r requirements.txt

ADD src/ /root/src/

WORKDIR /root

CMD [ "python", "src/main.py" ]