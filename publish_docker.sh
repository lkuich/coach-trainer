#!/bin/bash
docker build -t tensorflow-cpu .
docker tag tensorflow-cpu:latest 434908186061.dkr.ecr.us-west-1.amazonaws.com/tensorflow-cpu:latest
docker push 434908186061.dkr.ecr.us-west-1.amazonaws.com/tensorflow-cpu:latest
