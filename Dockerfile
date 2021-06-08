FROM python:3.9.2-slim-buster

RUN apt update 
RUN apt install -y python3-opencv

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ADD https://mgp.fra1.digitaloceanspaces.com/hound_model_vgg.pt hound_model_vgg.pt

CMD flask run --host 0.0.0.0 --port "$PORT"
