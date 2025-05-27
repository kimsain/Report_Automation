import os
import pandas as pd
import numpy as np
import sqlite3
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import sqlalchemy
from sqlalchemy import create_engine, text
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 데이터베이스 연결
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 데이터 추출 및 문서 생성 함수
def extract_data_from_db():
    """데이터베이스에서 데이터를 추출하여 문서 형태로 변환"""
    documents = []
    
    # 축제 통계 데이터 추출
    query = """
    SELECT 
        r.region_code,
        f.period_type,
        f.start_date,
        f.end_date,
        f.sales_amount,
        f.sales_increase_rate,
        f.main_business_type,
        f.visitors,
        f.visitor_increase_rate,
        f.main_age_group,
        f.main_time_period
    FROM festival_stats f
    JOIN regions r ON f.region_id = r.id
    """
    
    festival_df = pd.read_sql(query, engine)
    
    for _, row in festival_df.iterrows():
        content = f"""
        영역 구분: {row['region_code']}
        기간 구분: {row['period_type']}
        기간: {row['start_date']} ~ {row['end_date']}
        매출액(억): {row['sales_amount']}
        전주 대비 매출 증감률(%): {row['sales_increase_rate'] if not pd.isna(row['sales_increase_rate']) else '정보 없음'}
        주 매출 업종: {row['main_business_type']}
        방문인구(명): {row['visitors']}
        전주 대비 방문인구 증감률(%): {row['visitor_increase_rate'] if not pd.isna(row['visitor_increase_rate']) else '정보 없음'}
        주 방문 연령층: {row['main_age_group']}
        주 방문 시간대: {row['main_time_period']}
        """
        
        metadata = {
            "source": "festival_stats",
            "region_code": row['region_code'],
            "period_type": row['period_type']
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    # 성연령별 방문인구 데이터 추출
    query = """
    SELECT 
        r.region_code,
        p.gender,
        p.age_group,
        p.period_type,
        p.visitors
    FROM population_stats p
    JOIN regions r ON p.region_id = r.id
    """
    
    population_df = pd.read_sql(query, engine)
    
    # 영역 구분과 기간 구분별로 그룹화
    for (region_code, period_type), group in population_df.groupby(['region_code', 'period_type']):
        content = f"""
        영역 구분: {region_code}
        기간 구분: {period_type}
        성연령별 방문인구 정보:
        """
        
        for _, row in group.iterrows():
            content += f"\n{row['gender']} {row['age_group']} 방문인구(명): {row['visitors']}"
        
        metadata = {
            "source": "population_stats",
            "region_code": region_code,
            "period_type": period_type
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    # 업종별 매출 데이터 추출
    query = """
    SELECT 
        r.region_code,
        s.business_type,
        s.sales_amount
    FROM sales_stats s
    JOIN regions r ON s.region_id = r.id
    """
    
    sales_df = pd.read_sql(query, engine)
    
    # 영역 구분별로 그룹화
    for region_code, group in sales_df.groupby('region_code'):
        content = f"""
        영역 구분: {region_code}
        업종별 매출 정보:
        """
        
        for _, row in group.iterrows():
            content += f"\n{row['business_type']} 매출액: {row['sales_amount']}"
        
        metadata = {
            "source": "sales_stats",
            "region_code": region_code
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    # 시간대별 인구 및 매출 데이터 추출
    query = """
    SELECT 
        r.region_code,
        h.period_type,
        h.time_period,
        h.sales_amount,
        h.visitors
    FROM hourly_stats h
    JOIN regions r ON h.region_id = r.id
    """
    
    hourly_df = pd.read_sql(query, engine)
    
    # 영역 구분과 기간 구분별로 그룹화
    for (region_code, period_type), group in hourly_df.groupby(['region_code', 'period_type']):
        content = f"""
        영역 구분: {region_code}
        기간 구분: {period_type}
        시간대별 인구 및 매출 정보:
        """
        
        for _, row in group.iterrows():
            content += f"\n시간대 {row['time_period']} - 방문인구(명): {row['visitors']}, 매출액: {row['sales_amount']}"
        
        metadata = {
            "source": "hourly_stats",
            "region_code": region_code,
            "period_type": period_type
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    # 유입인구 데이터 추출
    query = """
    SELECT 
        r.region_code,
        i.inflow_region_code,
        i.visitors,
        i.inflow_type,
        i.region_name,
        i.province_name
    FROM inflow_stats i
    JOIN regions r ON i.region_id = r.id
    """
    
    inflow_df = pd.read_sql(query, engine)
    
    # 영역 구분별로 그룹화
    for region_code, group in inflow_df.groupby('region_code'):
        content = f"""
        영역 구분: {region_code}
        유입인구 정보:
        """
        
        for _, row in group.iterrows():
            content += f"\n{row['province_name']} {row['region_name']} ({row['inflow_type']}) - 방문인구: {row['visitors']}"
        
        metadata = {
            "source": "inflow_stats",
            "region_code": region_code
        }
        
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

# 문서 분할 함수
def split_documents(documents):
    """문서를 청크로 분할"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)

# 벡터 저장소 생성 함수
def create_vector_store(documents):
    """문서를 임베딩하여 벡터 저장소 생성"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # 벡터 저장소 저장
    vector_store.save_local("faiss_index")
    
    return vector_store

# 벡터 저장소 로드 함수
def load_vector_store():
    """저장된 벡터 저장소 로드"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("faiss_index", embeddings)
    
    return vector_store

# RAG 시스템 생성 함수
def create_rag_system(vector_store):
    """RAG 시스템 생성"""
    # 프롬프트 템플릿 정의
    template = """
    당신은 축제 데이터 분석 전문가입니다. 주어진 정보를 바탕으로 사용자의 질문에 정확하게 답변해 주세요.
    
    질문에 관련된 정보가 없는 경우, "제공된 데이터에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
    
    ### 관련 정보:
    {context}
    
    ### 질문:
    {question}
    
    ### 답변:
    """
    
    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # LLM 설정
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    # RAG 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 20}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

# 메인 함수
def setup_rag_system():
    """RAG 시스템 설정"""
    print("데이터베이스에서 데이터 추출 중...")
    documents = extract_data_from_db()
    print(f"추출된 문서 수: {len(documents)}")
    
    print("문서 분할 중...")
    chunks = split_documents(documents)
    print(f"생성된 청크 수: {len(chunks)}")
    
    print("벡터 저장소 생성 중...")
    vector_store = create_vector_store(chunks)
    print("벡터 저장소 생성 완료")
    
    print("RAG 시스템 설정 완료")
    return vector_store

if __name__ == "__main__":
    vector_store = setup_rag_system()  # 문서 추출 + 벡터 저장소 생성

    # RAG 시스템 실행 부분 (질문/응답)
    # qa_chain = create_rag_system(vector_store)

    # SQLDatabaseChain 구성
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    db = SQLDatabase(engine=engine)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

    question = "축제 기간 동안 가장 많이 방문한 연령층과 가장 적게 방문한 연령층을 알려줘"
    # response = qa_chain.invoke({"query": question})
    response = db_chain.invoke({"query": question})

    print("답변:", response["result"])
    print("관련 문서 수:", len(response["source_documents"]))
