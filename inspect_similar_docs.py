# inspect_similar_docs.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

# 환경 변수 로드
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# 연결 문자열 정의
connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 임베딩 모델 로딩
embeddings = OpenAIEmbeddings()

# 벡터스토어 연결
vector_store = PGVector(
    collection_name="embedding_store",
    connection_string=connection_string,
    embedding_function=embeddings
)

# 🔍 유사 문서 검색
query = "0지역의 축제 기간 동안 어떤 연령대가 가장 많이 방문했는가?"
docs = vector_store.similarity_search(query, k=100)

print(f"\n🔍 검색 쿼리: {query}")
print(f"\n🔢 유사 문서 개수: {len(docs)}\n")

for i, doc in enumerate(docs, 1):
    print(f"--- 문서 #{i} ---")
    print(doc.page_content)
    print()
