#!/bin/bash
$(aws ecr get-login --no-include-email --region us-east-1)
docker build -t tf2-gpu .
docker tag tf2-gpu:latest 434908186061.dkr.ecr.us-east-1.amazonaws.com/tf2-gpu:latest
docker push 434908186061.dkr.ecr.us-east-1.amazonaws.com/tf2-gpu:latest
