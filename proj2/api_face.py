from fastapi import FastAPI, File, UploadFile

# STEP 1 추론기에 필요한 패키지 가져옴
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2 추론기 객체 만듦 
face = FaceAnalysis()        # 모델을 자동으로 다운 받아주는 기능 있음
face.prepare(ctx_id=0, det_size=(640,640))   # ctx_id : GPU 크기 / det_size : CPU 크기 지정

# 등록기에 저장해 놓을 배열
target_face = []

app = FastAPI()

@app.post("/registFace/")
async def create_upload_file(file: UploadFile):

    content = await file.read()     # content를 open cv2 이미지로 변환해야 함

    # STEP 3 테스트 데이터 가져오기
    # img1 = cv2.imread("jihyun.jpg")
    # --> buf = file.open("jihyun.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # STEP 4 추론
    faces1 = face.get(img)
    assert len(faces1)==1

    # STEP 5 응용
    target_face.append(np.array(faces1[0].normed_embedding, dtype=np.float32))
    print(target_face)

    return {"result": len(faces1)}

@app.post("/compareFace/")
async def compareFace(file: UploadFile):

    content = await file.read()     # content를 open cv2 이미지로 변환해야 함

    # STEP 3 테스트 데이터 가져오기
    # img1 = cv2.imread("jihyun.jpg")
    # --> buf = file.open("jihyun.jpg")
    # --> img = cv2.imdecode(buf)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # STEP 4 추론
    faces1 = face.get(img)
    assert len(faces1)==1

    # STEP 5 응용
    test_face = np.array(faces1[0].normed_embedding, dtype=np.float32)
    sim = np.dot(target_face[0], test_face.T)

    return {"result": sim.item()}