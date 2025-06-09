# insight_generator.py
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

def generate_insight_scores(content_raw, crym, sido, sigungu, subject, query_text):
    """
    GPT 응답으로부터 인사이트와 URL을 추출하고, 점수 계산 및 최종 DataFrame 생성
    """
    insights_raw = content_raw.strip().split('\n\n')
    insight_texts, insight_urls = [], []

    for item in insights_raw:
        url_match = re.search(r'\((https?://.*?)\)', item)
        url = url_match.group(1) if url_match else None
        text_cleaned = re.sub(r'\[.*?\]\(.*?\)', '', item).replace("- ", "").strip()
        insight_texts.append(text_cleaned)
        insight_urls.append(url)

    # TF-IDF 계산
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(insight_texts)
    tfidf_scores = X.sum(axis=1).A1

    # GPT 점수 평가
    gpt_scores = []
    for insight in insight_texts:
        prompt = f"""
        다음 인사이트의 정보 밀도와 실용성을 평가하고, 1~10점으로 점수를 매기세요.
        인사이트: "{insight}"
        단, 점수만 숫자로 출력하세요.
        """
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                당신은 데이터 분석 전문가이며, 인사이트 평가 전문가입니다.
                                {crym}년 {sido} {sigungu}의 {subject}에 대한 인사이트를 평가합니다.
                                우선, 주어진 데이터 정보는 아래와 같으며, 인사이트를 평가해주세요

                                {query_text}
                               """
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=10
        )
        score_text = completion.choices[0].message.content.strip()
        try:
            score = float(score_text)
        except ValueError:
            score = 5.0
        gpt_scores.append(score)
        time.sleep(2)  # Rate limit 대응

    gpt_scores = np.array(gpt_scores)
    final_scores = 0.4 * tfidf_scores + 0.6 * gpt_scores

    # 결과 DataFrame 생성
    df_result = pd.DataFrame({
        'crym': [crym] * len(insight_texts),
        'sido': [sido] * len(insight_texts),
        'sigungu': [sigungu] * len(insight_texts),
        'subject': [subject] * len(insight_texts),
        'insight_no': list(range(1, len(insight_texts) + 1)),
        'insight': insight_texts,
        'tfidf_score': tfidf_scores,
        'gpt_score': gpt_scores,
        'final_score': final_scores,
        'url': insight_urls
    })

    df_result = df_result.sort_values(by='final_score', ascending=False)
    return df_result

# CSV 파일 읽기
file_path = os.path.join(os.getcwd(), "data")
file_name = "LLM_현황자동화텍스트.csv"
df = pd.read_csv(os.path.join(file_path, file_name), encoding='utf-8')

# 결과를 저장할 리스트
results = []

# 고유한 그룹 조합 추출
# group_keys = df[['crym', 'sido', 'sigungu', 'subject', 'slide']].drop_duplicates()
group_keys = df[['crym', 'sido', 'sigungu', 'subject']].drop_duplicates()

# OpenAI 클라이언트 초기화
client = OpenAI()

df_insight = pd.DataFrame()

for _, group in group_keys.iterrows():
    # crym, sido, sigungu, subject, slide = group['crym'], group['sido'], group['sigungu'], group['subject'], group['slide']
    crym, sido, sigungu, subject = group['crym'], group['sido'], group['sigungu'], group['subject']
    
    # 해당 그룹의 데이터 필터링
    # filtered_data = df[
    #     (df['crym'] == crym) &
    #     (df['sido'] == sido) &
    #     (df['sigungu'] == sigungu) &
    #     (df['subject'] == subject) &
    #     (df['slide'] == slide)
    # ]
    filtered_data = df[
        (df['crym'] == crym) &
        (df['sido'] == sido) &
        (df['sigungu'] == sigungu) &
        (df['subject'] == subject)
    ]
    
    # query_text 생성
    year = str(crym)[:4]
    month = str(crym)[4:6].lstrip("0")  # 앞의 0 제거
    query_text = f"{year}년 {month}월 {sido} {sigungu}의 현황 정보:\n"
    for _, row in filtered_data.iterrows():
        query_text += f" - {row['stmt']}\n"

    completion = client.chat.completions.create(
        model="gpt-4o-search-preview",
        web_search_options={},
        messages=[
            {
                "role": "system", 
                "content": 
                f"""
                    당신은 데이터 분석 전문가입니다.
                    주어진 지역 현황 정보를 바탕으로 해당 현상이 발생한 원인을 분석하고 설명하세요.

                    # 웹 검색을 통해 확인 가능한 {year}년 {month}월을 포함하여, 직전 3개월(과거 3개월) 정보만 활용하세요.
                    # 응답은 다음 형식을 따르세요:
                        - 불렛포인트 10개로 작성할 것.
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
        ]
    )

    temp_df = generate_insight_scores(
        content_raw=completion.choices[0].message.content,
        crym=crym,
        sido=sido,
        sigungu=sigungu,
        subject=subject,
        query_text=query_text,
    )

    df_insight = pd.concat([df_insight, temp_df], ignore_index=True)

# 결과 저장장
file_path = os.path.join(os.getcwd(), "data")
file_name = "insight_test.csv"
df_insight.to_csv(os.path.join(file_path, file_name), index=False, encoding='utf-8')