import json
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 데이터 로드
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

# 텍스트 정규화
def normalize_text(text):
    text = text.lower()  # 소문자 변환
    text = re.sub(r'\W', ' ', text)  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text)  # 불필요한 공백 제거
    return text.strip()

# 인사말 제거
def remove_greetings(text):
    greetings = ["안녕하세요", "안녕", "안녕하십니까", "고맙습니다", "감사합니다"]
    for greeting in greetings:
        text = text.replace(greeting, '')
    return text.strip()

# 한국어 불용어 목록 예시
korean_stopwords = [
    "의", "가", "이", "은", "는", "에", "와", "과", "를", "을", "이유", "다", "합니다", "합니다", "있습니다", "없습니다",
    "해도", "같은", "그", "그녀", "그들", "너", "나", "저", "우리", "당신", "한", "것", "이", "그런", "그것",
    # 필요한 불용어를 추가하세요.
]

# 텍스트 전처리
def preprocess_text(text):
    text = remove_greetings(text)  # 인사말 제거
    text = normalize_text(text)  # 텍스트 정규화
    tokens = word_tokenize(text)  # 토큰화
    stop_words = set(korean_stopwords)  # 직접 만든 한국어 불용어 목록
    filtered_tokens = [word for word in tokens if word not in stop_words]  # 불용어 제거
    return ' '.join(filtered_tokens)

# 데이터 프레임 생성
def create_dataframe(qa_data, legal_data):
    rows = []
    for entry in qa_data:
        question = preprocess_text(entry['question'])
        answer = preprocess_text(entry['answer'])
        rows.append({'question': question, 'answer': answer})

    for entry in legal_data:
        instruction = preprocess_text(entry['instruction'])
        output = preprocess_text(entry['output'])
        rows.append({'question': instruction, 'answer': output})

    return pd.DataFrame(rows)

# 데이터 전처리 파이프라인 실행
legalqa_data = load_data('C:/Users/bit/Ideaproject/ml-work-space/1rawdata/legalqa.jsonlines')  # 절대 경로로 수정
with open('C:/Users/bit/Ideaproject/ml-work-space/1rawdata/생활법령.json', 'r', encoding='utf-8') as f:  # 절대 경로로 수정
    생활법령_data = json.load(f)

df = create_dataframe(legalqa_data, 생활법령_data)
print(df.head())

df.to_csv('C:/Users/bit/Ideaproject/ml-work-space/2predata/combined_preprocessed_data.csv', index=False, encoding='utf-8-sig')  # 절대 경로로 수정
