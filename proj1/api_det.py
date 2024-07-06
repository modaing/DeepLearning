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
@app.post("/uploadfile/")   # http url !!!! 
async def create_upload_file(file: UploadFile):

    content = await file.read()

    # STEP 3: Load the input image.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary) # 참고 부분 https://stackoverflow.com/questions/73810377/how-to-save-an-uploaded-image-to-fastapi-using-python-imaging-library-pil
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img)) # 컬러 이미지의 경우

    # STEP 4: Classify the input image.
    detection_result = detector.detect(image)

    counts = len(detection_result.detections)
    object_list = []
    for detection in detection_result.detections:
        object_category = detection.categories[0].category_name
        object_list.append(object_category)

    print(detection_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    # top_category = classification_result.classifications[0].categories[0]
    # result = f"{top_category.category_name} ({top_category.score:.2f})"
    # return {"result" : result}
    return {"counts": counts,
            "object_list": object_list}