# STEP 1
from sentence_transformers import SentenceTransformer

# STEP 2
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# STEP 3
# The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

sentences1 = "체크 셔츠 좋아하시나봐요"
sentences2 = "제발 그만 입어"

# sentences1 = "나 살쪘지?"
# sentences2 = "날씬하다고 말해"

# STEP 4
# 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode(sentences)
# print(embeddings.shape)
# [3, 384]

embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)
print(embeddings1.shape)

# STEP 5
# 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

similarities1 = model.similarity(embeddings1, embeddings2)
print(similarities1)