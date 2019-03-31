sudo docker run -p 8501:8501 --mount type=bind,source=/opt/ml/model/export/flowers,target=/models/flowers -e MODEL_NAME=flowers -t tensorflow/serving &
