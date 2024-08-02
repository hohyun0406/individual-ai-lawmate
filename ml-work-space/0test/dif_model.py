from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 입력 문장을 정의합니다.
texts = ["We are very happy to show you the 🤗 Transformers library."]

# 문장을 인코딩하여 배치를 생성합니다.
pt_batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 모델에 입력을 주고 출력을 받습니다.
pt_outputs = model(**pt_batch)

# 결과를 출력합니다.
print(pt_outputs)
