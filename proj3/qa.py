# Question Classification

# STEP 1
from transformers import pipeline

# STEP 2
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")
# model 이 누구껀지 모르기때문에 허깅페이스에 로그인하라고 뜸
# but 내껄로 로그인하면 모델이 없으니까 정보안떠서 이럴 땐 만든 사람의 실수이므로 직접 찾아서(제목 / 앞에 있음) 적어주기

# STEP 3
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

# STEP 4
result = question_answerer(question=question, context=context)

# STEP 5
print(result)