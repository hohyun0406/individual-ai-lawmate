import os
from datetime import datetime
import pytz
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class Question(BaseModel):
    q_sentence: str

class Answer(BaseModel):
    answer_sentence: str

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

llm = None
tokenizer = None

@app.on_event("startup")
def startup_event():
    global tokenizer, llm

    print("Load LLM")
    model_id = "skt/kogpt2-base-v2"
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, device_map={"": 0}, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

@app.post("/generate", response_model=Answer)
async def generate(question: Question):
    KST = pytz.timezone('Asia/Seoul')
    print(datetime.now(KST).strftime("%Y/%m/%d %H:%M:%S"))

    q_sentence = question.q_sentence
    print(f"q_sentence: {q_sentence}")

    inputs = tokenizer(q_sentence, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    outputs = llm.generate(**inputs, max_length=50, num_return_sequences=1, do_sample=True)
    answer_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return Answer(answer_sentence=answer_sentence)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
