import requests
import json
import os
from PIL import Image
from io import BytesIO
import base64

def image_to_data_uri(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    with open(filename, 'rb') as f:
        data = f.read()
    return prefix + base64.b64encode(data).decode('utf-8')


url = 'http://127.0.0.1:5001/predictions'

# Open your image
source_uri = image_to_data_uri('source.jpg')
target_uri = image_to_data_uri('target.jpg')



# Prepare the payload
payload = json.dumps({
  "input": {
    "source": source_uri,
    "target": target_uri
  }
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
result = response.json()
content = result.get('output')
header, content = content.split("base64,", 1)
content = base64.b64decode(content)
with open("./output.jpg", "wb") as f:
    f.write(content)


