# 참고 https://fastapi.tiangolo.com/ko/tutorial/request-forms/
from fastapi import FastAPI, Form

# STEP 1 : 추론기를 가져오기 위한 패키지 가져옴
from transformers import pipeline

# STEP 2 : 추론기에 필요한 모델 가져옴 (객체 생성)
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

app = FastAPI()

@app.post("/textClassification/")
async def login(text: str = Form()):
    # STEP 3 : test data 입력
    # text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    # text = "현대바이오, '폴리탁셀' 코로나19 치료 가능성에 19% 급감"
    # text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다"

    # STEP 4 : 추론
    result = classifier(text)

    # STEP 5 : 후처리
    print(result)
    return {"result": result}