# rag_system_v2.py (수정본)
# 결과: embedding_store DB가 존재하고, 매번 생성 필요 없음

import os
import re
import time
import torch
import faiss
import psycopg2

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

# 환경 변수 로드
load_dotenv()

# DB 연결 정보
# DB_NAME = os.getenv("DB_NAME")
# DB_USER = os.getenv("DB_USER")
# DB_PASSWORD = os.getenv("DB_PASSWORD")
# DB_HOST = os.getenv("DB_HOST")
# DB_PORT = os.getenv("DB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DB 연결
# conn = psycopg2.connect(
#     dbname=DB_NAME,
#     user=DB_USER,
#     password=DB_PASSWORD,
#     host=DB_HOST,
#     port=DB_PORT
# )
# cursor = conn.cursor()

query_text = """
2025년 2월 현황 정보:
 - 수원시의 당월 전체 신규 가맹점 수는 전월 대비 감소
 - 휴·폐업 가맹점은 증가하여 전반적으로 사업자 유입 흐름이 감소
 - 일반의원 업종에서 09-11시간대에 최대 매출액이 발생
 - 편의점 업종에서 19-21시간대에 최다 매출건수가 발생
"""


client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-search-preview",
    web_search_options={},
    messages=[
        {
            "role": "system", 
            "content": 
            """
                당신은 데이터 분석 전문가입니다.
                주어진 지역 현황 정보를 바탕으로 해당 현상이 발생한 원인을 분석하고 설명하세요.

                # 웹 검색을 통해 확인 가능한 2025년 2월을 포함하여, 직전 3개월(과거 3개월) 정보만 활용하세요.
                # 응답은 다음 형식을 따르세요:
                    - 불렛포인트 10개로만 작성할 것.
                    - 각 포인트는 2~3문장으로 구성할 것.
                    - 각 원인 설명에는 참고한 출처(URL 포함)를 명시할 것.
                    - 불렛포인트외 다른 문장은 응답에 포함하지 말 것.
                # 응답은 중립적이고 객관적인 톤을 유지할 것.
            """
        },
        {
            "role": "user",
            "content": query_text,
        }
    ],
    # temperature=0.5,      # 출력의 창의성/무작위성을 조절. 값 범위: 0 ~ 2
    # max_tokens=500,       # 생성되는 응답의 최대 길이 (토큰 수 기준).
    # top_p=1,              # 샘플링을 위한 확률적 선택을 조절. 값 범위: 0 ~ 1
    # frequency_penalty=0,  # 반복되는 단어 사용을 줄이기 위한 패널티. 값 범위: -2 ~ 2
    # presence_penalty=0,   # 새로운 주제의 단어 사용을 장려하기 위한 패널티. 값 범위: -2 ~ 2
    # stop=None,            # 응답 종료를 위한 토큰
    # n=1,                  # 생성할 응답의 개수
    # stream=False,         # 스트리밍 응답 여부
    # user="user_id",       # 사용자 ID
    # logit_bias={},        # 특정 토큰의 생성 확률을 조정하기 위한 bias
)
                # 웹 검색을 통해 확인 가능한 2025년 1월과 2월의 정보만 활용하세요.
# print(completion.choices[0].message.content)

# 1. 인사이트 본문과 URL 분리
content_raw = completion.choices[0].message.content
insights_raw = content_raw.strip().split('\n\n')

insight_texts = []
insight_urls = []

for item in insights_raw:
    # 링크 추출
    url_match = re.search(r'\((https?://.*?)\)', item)
    url = url_match.group(1) if url_match else None
    
    # 본문 텍스트에서 링크 부분 삭제
    text_cleaned = re.sub(r'\[.*?\]\(.*?\)', '', item)  # [텍스트](링크) 제거
    text_cleaned = text_cleaned.replace("- ", "").strip()
    
    insight_texts.append(text_cleaned)
    insight_urls.append(url)

# 3. TF-IDF 정보 밀도 계산
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(insight_texts)
tfidf_scores = X.sum(axis=1).A1  # 각 문서별 TF-IDF 총합

# 4. GPT 평가 점수 API 호출
client = OpenAI()
gpt_scores = []

for insight in insight_texts:
    evaluation_prompt = f"""
    다음 인사이트의 정보 밀도와 실용성을 평가하고, 1~10점으로 점수를 매기세요.
    인사이트: "{insight}"
    단, 점수만 숫자로 출력하세요.
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 데이터 분석 전문가이며, 인사이트 평가 전문가입니다."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0.3,
        max_tokens=10
    )
    
    score_text = completion.choices[0].message.content.strip()
    
    # 점수 숫자형 변환
    try:
        score = float(score_text)
    except ValueError:
        score = 5.0  # 파싱 실패 시 중간값
    
    gpt_scores.append(score)
    time.sleep(2)  # API rate limit 대응

# 4. 가중 평균 계산 (TF-IDF 40%, GPT 평가 60%)
gpt_scores = np.array(gpt_scores)
final_scores = 0.4 * tfidf_scores + 0.6 * gpt_scores

# 5. DataFrame으로 정리
df = pd.DataFrame({
    'insight': insight_texts,
    'url': insight_urls,
    'tfidf_score': tfidf_scores,
    'gpt_score': gpt_scores,
    'final_score': final_scores
})

# 6. 상위 3개 도출
top_insights = df.sort_values(by='final_score', ascending=False).head(3)

# 결과 출력
file_path = os.path.join(os.getcwd(), "data")
file_name = "insight_test.csv"
df.to_csv(os.path.join(file_path, file_name), index=False, encoding='utf-8')