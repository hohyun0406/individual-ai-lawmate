import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, DataCollatorWithPadding, Trainer
from datasets import load_dataset
import torch

# 데이터셋 로드
dataset = load_dataset('csv', data_files='C:/Users/bit/Ideaproject/individual-ml-lawmate/ml-work-space/4model/train_data.csv')

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
        user_input=data_point['question'],
        midjourney_prompt=data_point['answer']
    )

# 프롬프트 생성 및 토큰화
dataset = dataset['train'].add_column('prompt', [generate_prompt(d) for d in dataset['train']])
dataset = dataset.map(lambda samples: tokenizer(samples['prompt'], padding='max_length', truncation=True), batched=True)

# start_positions와 end_positions을 데이터셋에 추가
# Note: 이 부분은 실제 데이터 구조에 따라 조정이 필요합니다.
# 여기서는 예시로 간단히 'answer' 컬럼의 시작과 끝 위치를 사용합니다.
dataset = dataset.map(lambda samples: {'start_positions': samples['answer'].find(samples['answer']),
                                       'end_positions': samples['answer'].find(samples['answer']) + len(samples['answer'])}, batched=True)

# 모델 준비 (Question Answering용 모델 로드)
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

# 데이터 컬레이터 설정
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Custom Trainer 설정
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_positions = inputs["start_positions"]
        end_positions = inputs["end_positions"]

        # Compute the loss using CrossEntropyLoss
        loss_fct = torch.nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        return (total_loss, outputs) if return_outputs else total_loss

# Trainer 설정
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=500,
    max_steps=2000,
    learning_rate=2e-5,
    logging_steps=10,
    output_dir="outputs",
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=100
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# 모델 훈련
trainer.train()

# 모델 저장
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
