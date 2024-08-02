from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI

app = FastAPI()

# 모델 로드
model_name = "kfkas/Legal-Llama-2-ko-7b-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

@app.post("/generate")
async def generate_text_api(prompt: str):
    result = generate_text(prompt)
    return {"generated_text": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
