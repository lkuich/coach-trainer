import requests
import base64
from matplotlib.image import imread
import json
import numpy as np

#encoded_image = None
#with open("/home/loren/rose-small.jpg", "rb") as image_file:
#    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
#print(encoded_image)


img = imread('/home/loren/rose-small.jpg')
print(img.shape)
img = np.stack(img, axis=0)
print(img.shape)
#print(str(img))

#img = [[[95,42,34],[103,112,83,[71,134,87],[33,81,41],[143,120,10,[152,60,71],[86,0,6],[125,81,80],[104,105,91,[63,54,45],[103,41,52],[117,0,30]],[[105,8,15],[127,77,68],[117,111,87],[113,94,77],[144,69,73],[158,35,56],[130,13,32],[124,54,62],[86,61,56],[53,28,24],[86,29,38],[94,2,23]],[[117,0,3],[149,39,50],[126,35,42],[132,30,41],[122,0,12],[146,0,22],[148,8,35],[134,31,48],[94,24,32],[69,10,16],[84,23,30],[76,15,23]],[[115,3,17],[154,36,52],[145,13,34],[151,3,29],[164,7,38],[157,2,33],[121,0,5],[122,0,16],[135,17,39],[118,13,30],[111,27,40],[91,32,36]],[[147,107,97],[134,55,58],[144,21,39],[117,0,0],[131,0,18],[122,0,19],[104,0,2],[143,5,30],[167,14,44],[156,11,40],[132,19,37],[110,41,44]]]

#object_for_api = {"instances": [{img}]}

#response = requests.post(url='http://localhost:8501/v1/models/flowers/versions/1:predict', json=object_for_api)

data = json.dumps({"signature_name": "serving_default", "instances": img})
#print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/flowers/versions/1:predict', data=data, headers=headers)
#predictions = json.loads(json_response.text)['predictions']

print(json_response.text)
