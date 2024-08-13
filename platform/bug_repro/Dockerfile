# Specify BROKER_URL and QUEUE when running
FROM python:3.13-rc-bullseye

RUN apt-get update && \
    apt-get install -y curl ca-certificates amqp-tools \
       --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
RUN pip install wandb
COPY ./main.py /main.py

ENV WANDB_API_KEY=local-017febd67d206c09d4bda0f66d949f02412e7a92
ENV WANDB_BASE_URL=http://k8s-default-appingre-6226b38256-166782791.us-east-1.elb.amazonaws.com

CMD  /usr/bin/amqp-consume --url=$BROKER_URL -q $QUEUE -c 1 /main.py