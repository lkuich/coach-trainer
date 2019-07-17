#!/bin/bash
$(aws ecr get-login --no-include-email --region us-east-1)
docker build -t tensorflow-gpu .
docker tag tensorflow-gpu:latest 434908186061.dkr.ecr.us-east-1.amazonaws.com/tensorflow-gpu:latest
docker push 434908186061.dkr.ecr.us-east-1.amazonaws.com/tensorflow-gpu:latest
