from transformers import pipeline


classifier = pipeline("sentiment-analysis")

result = classifier("We are very happy to show you the 🤗 Transformers library.")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
