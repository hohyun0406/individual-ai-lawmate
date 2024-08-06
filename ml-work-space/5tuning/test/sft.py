import transformers
from trl import SFTTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

# 데이터셋 로드
dataset = load_dataset('csv', data_files='C:/Users/bit/Ideaproject/individual-ml-lawmate/ml-work-space/4model/train_data.csv')
print(dataset)

# BERT 모델의 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 사용자 입력과 모델 응답을 사용하여 프롬프트 생성
def generate_prompt(data_point):
    prompt_template = '''
    <start_of_turn>user
    {user_input}

    <end_of_turn>\n<start_of_turn>model
    {midjourney_prompt}
    '''
    return prompt_template.format(
        user_input=data_point['question'],  # 질문으로 변경
        midjourney_prompt=data_point['answer']  # 답변으로 변경
    )

# 프롬프트 생성 및 토큰화
dataset = dataset['train'].add_column('prompt', [generate_prompt(d) for d in dataset['train']])
dataset = dataset.map(lambda samples: tokenizer(samples['prompt'], padding='max_length', truncation=True), batched=True)

# 모델 준비 (Sequence Classification용 모델 로드)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)  # num_labels는 1로 설정
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 데이터 컬레이터 설정
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer 설정
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # 배치 크기를 8로 조정
    gradient_accumulation_steps=2,  # 그래디언트 축적 스텝 조정
    warmup_steps=500,  # warmup_steps 조정
    max_steps=2000,  # 최대 스텝 수 조정
    learning_rate=2e-5,  # 학습률 조정
    logging_steps=10,  # 로깅 스텝
    output_dir="outputs",  # 출력 디렉토리
    optim="adamw_hf",  # 옵티마이저 설정
    save_strategy="steps",  # 저장 전략
    save_steps=100  # 저장 스텝 조정
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # 훈련 데이터셋 사용
    data_collator=data_collator,  # 데이터 컬레이터 추가
    peft_config=LoraConfig(),  # LoRA 설정
)

# 모델 훈련
trainer.train()

# 모델 저장
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
