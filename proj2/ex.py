# STEP 1 추론기에 필요한 패키지 가져옴
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# STEP 2 추론기 객체 만듦 
app = FaceAnalysis()        # 모델을 자동으로 다운 받아주는 기능 있음
app.prepare(ctx_id=0, det_size=(640,640))   # ctx_id : GPU 크기 / det_size : CPU 크기 지정

# STEP 3 테스트 데이터 가져오기
# from insightface.data import get_image as ins_get_image     # SAMPLE TEST DATA 가져오는 도구
# img = ins_get_image('t1')

# local file test data 가져오기
img1 = cv2.imread("jihyun.jpg")
img2 = cv2.imread("jun.jpg")

# STEP 4 추론
# faces = app.get(img)
# assert len(faces)==6

faces1 = app.get(img1)
faces2 = app.get(img2)
assert len(faces1)==1   # 얼굴 1개니까 1
assert len(faces2)==1

print(faces1[0])

# STEP 5 응용
rimg = app.draw_on(img1, faces1)
cv2.imwrite("./jihyun_output.jpg", rimg)


# then print all-to-all face similarity
# feats = []
# for face in faces:
#     feats.append(face.normed_embedding)

feat1 = np.array(faces1[0].normed_embedding, dtype=np.float32)
feat2 = np.array(faces1[0].normed_embedding, dtype=np.float32)  
# faces1[0]의 이미지를 꺼내서 normed_embedding으로 변환, 데이터 타입은 dtype=np.float32으로 바꾸고 np.array 넘파이 배열(행렬)로 변환한다.
sim = np.dot(feat1, feat2.T)
# np.dot 을 곱하면 두개의 행렬곱을 나타낸다..? np.dot으로 vectorizing 효과를 본다.
print(sim)

