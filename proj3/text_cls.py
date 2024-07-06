# Text Classification

# STEP 1 : 추론기를 가져오기 위한 패키지 가져옴
from transformers import pipeline

# STEP 2 : 추론기에 필요한 모델 가져옴 (객체 생성)
# 추론기에 꼭 들어가야될 모델 정보를 pipeline이 모두 담고 있음 
# pipeline 도 실행하면 모델을 자동으로 받아주는 기능 있음
# model="누구의 모델/어떤 모델"
# 문장의 긍부정을 분류하는 모델
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# STEP 3 : test data 입력
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
# text = "현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급감"
text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다"
# => 위의 text는 입장에 따라서 해석(긍부정)이 다르므로, 비정형 데이터이다.
#    이 text는 positive로 나오는데 삼성전자의 입장에서 negative로 나오도록 해결하려면 
#    Question Classification 모델을 함께 돌려서 "어떤 회사가 더 성장하고 있어? 점유율이 더 높아?"라는 질문으로 구분 가능함

# STEP 4 : 추론
result = classifier(text)

# STEP 5 : 후처리
print(result)