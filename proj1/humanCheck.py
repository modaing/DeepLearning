from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. // 패키지 가져오기
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path='models\det\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

app = FastAPI()

from PIL import Image
import numpy as np
import io

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):

    content = await file.read()

    # STEP 3: Load the input image.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary) 
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image.
    detection_result = detector.detect(image)

    counts = len(detection_result.detections)
    object_list = []
    for detection in detection_result.detections:
        object_category = detection.categories[0].category_name
        object_list.append(object_category)

    print(detection_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    if counts is None:
        return {
            "result": "없음",
            "counts": counts,
            "who": object_list
            }
    else:
        return {
            "result": "있음",
            "counts": counts,
            "who": object_list
            }