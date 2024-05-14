import requests
import json
import subprocess
import os
import time
import base64
from base64 import b64decode
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware



app = FastAPI(title="facefusion",version='0.0.1')


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])  

class Parameters(BaseModel):
    debug: bool = Field(title="Debug", type="boolean", description="provide debugging output in logs", default=True)
    source: str = Field(title="Source image", type="string", description="Source image for face swap")
    target: str = Field(title="Target image", type="string", description="Target image for face swap")
    quality: int = Field(title="Quality", type="integer", description="Quality of the output image(from 0 to 100)", default=80)
    

class InputData(BaseModel):
    input: Parameters = Field(title="Input")

def image_to_data_uri(filename):
    ext = filename.split('.')[-1]
    prefix = f'data:image/{ext};base64,'
    with open(filename, 'rb') as f:
        data = f.read()
    return prefix + base64.b64encode(data).decode('utf-8')

@app.get("/health-check")
def health_check():
    return {"status": "True"}

@app.post("/predictions")
def aigic(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. Input Data processing...')

    # Prepare the data 
    source_uri = inputdata.input.source
    header, source_encoded = source_uri.split("base64,", 1)
    source_decoded = b64decode(source_encoded)
    with open("/bin/source.jpg", "wb") as f:
        f.write(source_decoded)
    
    if os.path.exists('/bin/source.jpg'):
        print('saved source image')
    target_uri = inputdata.input.target
    header, target_encoded = target_uri.split("base64,", 1)
    target_decoded = b64decode(target_encoded)
    with open("/bin/target.jpg", "wb") as f:
        f.write(target_decoded)
    if os.path.exists('/bin/target.jpg'):
        print('saved target image')

    quality = str(inputdata.input.quality)

    print('2. Running the model...')
    # Run the model
    subprocess.run(['python', 'run.py', '--frame-processors', 'face_swapper', 'face_enhancer', '-s', '/bin/source.jpg', '-t', '/bin/target.jpg', '-o', '/bin/output.jpg', '--output-image-quality', quality, '--headless', '--skip-download'])

    while not os.path.exists('/bin/output.jpg'):
        print('Waiting for the output image...')
        time.sleep(1)

    # Read the output image
    output = image_to_data_uri('/bin/output.jpg')
    # Remove the images
    os.remove('/bin/source.jpg')
    os.remove('/bin/target.jpg')
    os.remove('/bin/output.jpg')

    print('3. Output')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')

    if len(output) == 0:
        return {
        #"code": 400,
        #"msg": "error",
        "output": "No output"
        }


    return {
    #"code": 200,
    #"msg": "success",
    "output": output 
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=5001)