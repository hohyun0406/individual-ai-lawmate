# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M",
  messages=[
    {"role": "system", "content": "당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다. You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner."},
    {"role": "user", "content": "폭행을 당했어. 어떻게 해야할까?"}
  ],
  temperature=0.7,
)

print(completion.choices[0].message)