# 푸틴 행동 추적기
# who : putin?? russia / 독재자/ 전쟁중인 나라 대통령 / 사람
# where : 북한과 베트남 방문, 쿠바 방문
# when : 2022년 2월 우크라이나와 러시아의 전쟁 후 (현재 진행)
# result : 전쟁 발생 가능성이 높은 나라 (전쟁 발생 주시해야 할 국가) (내 의견)
# =>> 이런식으로 누가(유명인) 언제 어디를 다녀왔는지를 추적할 수 있다. (멘토님 의견)
# 기사 : https://www.econotimes.com/Why-Russia-still-has-friends-on-the-world-stage-1680793

# STEP 1
from transformers import pipeline

# STEP 2
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

# STEP 3
title = "Why Russia still has friends on the world stage"
body = "Russian president Vladimir Putin’s recent visits to both North Korea and Vietnam received significant attention in the western media.\
        So, too, did a recent visit by Russian warships to Cuba.\
        Before the outbreak of the full-blown war in Ukraine in February 2022, such visits would have likely received much less attention.\
        Now, they come amid western attempts to isolate Russia on the world stage.\
        However, it seems these efforts have had little effect in undermining many of Russia’s international relationships."

question = "Where has Putin visited recently??"

# STEP 4
result1 = classifier(title)
result2 = classifier(body)

# question_answerer(question=question, context=title)
result3 = question_answerer(question=question, context=body)

result = summarizer(body)

# STEP 5
print(result1)
print(result2)

print(result3)

print(result)