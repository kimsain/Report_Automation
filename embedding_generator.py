# -*- coding: utf-8 -*-
# @Time    : 2025/04/21 12:36
# @Author  : Sungjae Park
# @Email   : spark9504@gmail.com
# @File    : embedding_generator.py
# @Description : 임베딩 생성기
# @Version : 1.0
"""
# 임베딩 생성기
이 스크립트는 PostgreSQL 데이터베이스에서 데이터를 추출하고, 주어진 문장에 대한 임베딩을 생성하여 저장하는 기능을 수행합니다.
- 데이터베이스 연결
- 테이블 생성
- 데이터
- 문장

- 임베딩 생성
- 임베딩 저장
"""
# 필요한 라이브러리 임포트       

import os
import json
import psycopg2
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

# 환경 변수 로드
load_dotenv()

# DB 연결 정보
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# DB 연결
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
cursor = conn.cursor()

# 테이블 생성 (없을 시)
cursor.execute("""
CREATE TABLE IF NOT EXISTS embedding_store (
    id SERIAL PRIMARY KEY,
    table_name TEXT,
    original_text TEXT,
    metadata JSONB,
    embedding VECTOR(768)
);
""")
conn.commit()

# 모델 로드
model_name = "BM-K/KoSimCSE-roberta-multitask"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# GPU 사용 가능 시 이동
if torch.cuda.is_available():
    model = model.to("cuda")

# 문장 임베딩 함수 (CLS pooling)
def get_embedding(text: str) -> list:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding.tolist()

# 대상 테이블 정의 및 데이터 추출 쿼리
tables = {
    "festival_stats": "SELECT r.region_code, f.period_type, f.visitors, f.sales_amount, f.main_age_group, f.main_business_type FROM festival_stats f JOIN regions r ON f.region_id = r.id",
    "population_stats": "SELECT r.region_code, p.gender, p.age_group, p.period_type, p.visitors FROM population_stats p JOIN regions r ON p.region_id = r.id",
    "sales_stats": "SELECT r.region_code, s.business_type, s.sales_amount FROM sales_stats s JOIN regions r ON s.region_id = r.id",
    "hourly_stats": "SELECT r.region_code, h.period_type, h.time_period, h.visitors, h.sales_amount FROM hourly_stats h JOIN regions r ON h.region_id = r.id",
    "inflow_stats": "SELECT r.region_code, i.inflow_region_code, i.visitors, i.inflow_type, i.region_name, i.province_name FROM inflow_stats i JOIN regions r ON i.region_id = r.id"
}

# 각 테이블에서 데이터 추출 및 임베딩 저장
for table, query in tables.items():
    print(f"\n\U0001F4C2 {table} 테이블에서 문장 생성 및 임베딩 중...")
    cursor.execute(query)
    rows = cursor.fetchall()

    for row in tqdm(rows):
        # 문장 생성 로직 (단순화 또는 커스터마이즈 가능)
        if table == "festival_stats":
            text = f"\n    영역 {row[0]}의 {row[1]} 기간:\n    방문자 수 {row[2]}명, 매출 {row[3]}억,\n    주요 연령층 {row[4]}, 주 매출 업종: {row[5]}"
            metadata = {"region_code": row[0], "period_type": row[1]}

        elif table == "population_stats":
            text = f"{row[0]} 지역 {row[1]} {row[2]}의 {row[3]} 방문객 수는 {row[4]}명입니다."
            metadata = {"region_code": row[0], "gender": row[1], "age_group": row[2], "period_type": row[3]}

        elif table == "sales_stats":
            text = f"{row[0]} 지역의 {row[1]} 업종 매출은 {row[2]}입니다."
            metadata = {"region_code": row[0], "business_type": row[1]}

        elif table == "hourly_stats":
            text = f"{row[0]} 지역 {row[1]} 시간대({row[2]}) 방문객 수는 {row[3]}명이고 매출은 {row[4]}입니다."
            metadata = {"region_code": row[0], "period_type": row[1], "time_period": row[2]}

        elif table == "inflow_stats":
            text = f"{row[5]} {row[4]}에서 유입된 인구는 {row[2]}명이며 유입유형은 {row[3]}입니다."
            metadata = {"region_code": row[0], "inflow_type": row[3], "region_name": row[4]}

        else:
            continue

        # 임베딩 생성 및 저장
        embedding = get_embedding(text)
        cursor.execute("""
            INSERT INTO embedding_store (table_name, original_text, metadata, embedding)
            VALUES (%s, %s, %s, %s)
        """, (table, text, json.dumps(metadata), embedding))

    conn.commit()
    print(f"✅ {table} 테이블 완료. 저장된 문장 수: {len(rows)}")

print("\n\U0001F389 모든 테이블 임베딩 완료")
conn.close()
