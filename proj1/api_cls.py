from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. // 패키지 가져오기
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. // 추론기 객체 만들기
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite')
# mediapipe 에서는 baseoption에 모델 옵션 상대 경로 맞춰야 함 
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=1)
# 옵션의 결과가 항상 1개만 나오도록 설정해줌
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

# FAST API 만들때는 이 부분 날려버리기 **암기
# @app.post("/files/")
# async def create_file(file: bytes = File()):
#     return {"file_size": len(file)}

from PIL import Image
import numpy as np
import io
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read() # 위의 file: bytes 와 content 가 같다, 비동기로 서버에서 받은 이미지 파일 읽어옴

    # content -> jpg 파일인데... http 통신에서는 파일이 character type 왔다갔다함.
    # 1. text ->binary      : io.BytesIO(text)
    # 2. binary -> PIL Image

    # STEP 3: Load the input image.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary) # 참고 부분 https://stackoverflow.com/questions/73810377/how-to-save-an-uploaded-image-to-fastapi-using-python-imaging-library-pil
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img)) # 컬러 이미지의 경우
    # 흑백 이미지 같은 경우 ImageFormat.GRAY 

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(image)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    return {"result" : result}