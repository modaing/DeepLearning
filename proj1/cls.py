import urllib.request

IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg', 'bird.jpg', 'cake.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

# https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg
# https://storage.googleapis.com/mediapipe-tasks/image_classifier/cat.jpg

# import cv2
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img)
#   cv2.imshow("test", img) 
#   cv2.waitKey(0)


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)





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


# STEP 3: Load the input image.
# image = mp.Image.create_from_file(IMAGE_FILENAMES[0])
image = mp.Image.create_from_file('cake.jpg') 
# 둘중 하나로 데이터 가져오기 가능

# STEP 4: Classify the input image.
classification_result = classifier.classify(image)
# 변경 안함. 추론하고 추론 결과 가져옴
print(classification_result)

# STEP 5: Process the classification result. In this case, visualize it.
top_category = classification_result.classifications[0].categories[0]
# 제일 높은 위치에 있는 결과를 가져옴
print(f"{top_category.category_name} ({top_category.score:.2f})")
# step 4에서 추론 완료해서 없어두 됨
# 응용 => 사용자에게 어떻게 보여줄거냐(어플리케이션 영역), console.log로 찍기, 이미지로 보여주기 등등
