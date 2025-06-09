"""
LangChain 기반 데이터베이스 관리 모듈

이 모듈은 LangChain을 활용한 보고서 자동화 시스템의 데이터베이스 스키마 생성 및 관리 기능을 제공합니다.
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_config import LangChainConfig

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = LangChainConfig.OPENAI_API_KEY

class LangChainDatabaseManager:
    """
    LangChain 기반 데이터베이스 관리 클래스
    """
    
    def __init__(self):
        """데이터베이스 매니저 초기화"""
        self.db_params = LangChainConfig.get_db_connection_params()
        self.embeddings = OpenAIEmbeddings(
            model=LangChainConfig.OPENAI_EMBEDDING_MODEL,
            openai_api_key=LangChainConfig.OPENAI_API_KEY
        )
    
    def connect_to_db(self):
        """데이터베이스 연결"""
        return psycopg2.connect(**self.db_params)
    
    def create_database_schema(self):
        """데이터베이스 스키마 생성"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # PGVector 확장 활성화
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 지역 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS regions (
                id SERIAL PRIMARY KEY,
                region_code INTEGER UNIQUE NOT NULL,
                region_name VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 축제 통계 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS festival_stats (
                id SERIAL PRIMARY KEY,
                region_id INTEGER REFERENCES regions(id),
                period_type VARCHAR(20) NOT NULL,
                start_date VARCHAR(20),
                end_date VARCHAR(20),
                sales_amount FLOAT,
                sales_increase_rate FLOAT,
                main_business_type VARCHAR(50),
                visitors FLOAT,
                visitor_increase_rate FLOAT,
                main_age_group VARCHAR(20),
                main_time_period VARCHAR(20),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 인구 통계 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS population_stats (
                id SERIAL PRIMARY KEY,
                region_id INTEGER REFERENCES regions(id),
                gender VARCHAR(10),
                age_group VARCHAR(10),
                period_type VARCHAR(20),
                visitors FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 매출 통계 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sales_stats (
                id SERIAL PRIMARY KEY,
                region_id INTEGER REFERENCES regions(id),
                business_type VARCHAR(50),
                sales_amount FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 시간대별 통계 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS hourly_stats (
                id SERIAL PRIMARY KEY,
                region_id INTEGER REFERENCES regions(id),
                period_type VARCHAR(20),
                time_period VARCHAR(10),
                sales_amount FLOAT,
                visitors FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 유입 통계 테이블
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS inflow_stats (
                id SERIAL PRIMARY KEY,
                region_id INTEGER REFERENCES regions(id),
                inflow_region_code VARCHAR(20),
                visitors FLOAT,
                inflow_type VARCHAR(10),
                region_code VARCHAR(20),
                region_name VARCHAR(50),
                province_name VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 임베딩 저장소 테이블
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS embedding_store (
                id SERIAL PRIMARY KEY,
                table_name TEXT NOT NULL,
                original_text TEXT NOT NULL,
                metadata JSONB,
                embedding VECTOR({LangChainConfig.VECTOR_DIMENSION}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            
            # 인덱스 생성
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_store_table_name ON embedding_store(table_name);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embedding_store_embedding ON embedding_store USING ivfflat (embedding vector_cosine_ops);")
            
            conn.commit()
            print("데이터베이스 스키마가 성공적으로 생성되었습니다.")
            
        except Exception as e:
            print(f"데이터베이스 스키마 생성 중 오류 발생: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def load_csv_data(self, data_dir: str = "./data"):
        """CSV 데이터를 데이터베이스에 로드"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # 축제분석_현황판.csv 로드
            festival_file = os.path.join(data_dir, '축제분석_현황판.csv')
            if os.path.exists(festival_file):
                festival_df = pd.read_csv(festival_file)
                self._load_festival_data(cursor, festival_df)
            
            # 성연령별_방문인구.csv 로드
            population_file = os.path.join(data_dir, '성연령별_방문인구.csv')
            if os.path.exists(population_file):
                population_df = pd.read_csv(population_file)
                self._load_population_data(cursor, population_df)
            
            # 업종별 매출.csv 로드
            sales_file = os.path.join(data_dir, '업종별 매출.csv')
            if os.path.exists(sales_file):
                sales_df = pd.read_csv(sales_file)
                self._load_sales_data(cursor, sales_df)
            
            # 시간대별 인구 및 매출.csv 로드
            hourly_file = os.path.join(data_dir, '시간대별 인구 및 매출.csv')
            if os.path.exists(hourly_file):
                hourly_df = pd.read_csv(hourly_file)
                self._load_hourly_data(cursor, hourly_df)
            
            # 유입인구.csv 로드
            inflow_file = os.path.join(data_dir, '유입인구.csv')
            if os.path.exists(inflow_file):
                inflow_df = pd.read_csv(inflow_file)
                self._load_inflow_data(cursor, inflow_df)
            
            conn.commit()
            print("CSV 데이터가 성공적으로 로드되었습니다.")
            
        except Exception as e:
            print(f"CSV 데이터 로드 중 오류 발생: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def _load_festival_data(self, cursor, df):
        """축제 데이터 로드"""
        # 지역 코드 추출 및 regions 테이블에 삽입
        if '영역 구분' in df.columns:
            region_codes = df['영역 구분'].unique()
            for code in region_codes:
                cursor.execute("""
                    INSERT INTO regions (region_code) 
                    VALUES (%s) 
                    ON CONFLICT (region_code) DO NOTHING
                """, (int(code),))
        
        # 축제 통계 데이터 삽입
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO festival_stats 
                (region_id, period_type, start_date, end_date, sales_amount, 
                 sales_increase_rate, main_business_type, visitors, 
                 visitor_increase_rate, main_age_group, main_time_period)
                SELECT r.id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                FROM regions r WHERE r.region_code = %s
            """, (
                row.get('구분'), row.get('시작'), row.get('종료'),
                row.get('매출액(억)'), row.get('전주 대비 증감률(%)_x'),
                row.get('주 매출 업종'), row.get('방문인구(명)'),
                row.get('전주 대비 증감률(%)_y'), row.get('주 방문 연령층'),
                row.get('주 방문 시간대'), int(row.get('영역 구분'))
            ))
    
    def _load_population_data(self, cursor, df):
        """인구 데이터 로드"""
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO population_stats 
                (region_id, gender, age_group, period_type, visitors)
                SELECT r.id, %s, %s, %s, %s
                FROM regions r WHERE r.region_code = %s
            """, (
                row.get('성별'), row.get('연령대'), row.get('구분'),
                row.get('방문인구(명)'), int(row.get('영역 구분'))
            ))
    
    def _load_sales_data(self, cursor, df):
        """매출 데이터 로드"""
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO sales_stats 
                (region_id, business_type, sales_amount)
                SELECT r.id, %s, %s
                FROM regions r WHERE r.region_code = %s
            """, (
                row.get('업종명'), row.get('이용금액'),
                int(row.get('영역 구분'))
            ))
    
    def _load_hourly_data(self, cursor, df):
        """시간대별 데이터 로드"""
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO hourly_stats 
                (region_id, period_type, time_period, sales_amount, visitors)
                SELECT r.id, %s, %s, %s, %s
                FROM regions r WHERE r.region_code = %s
            """, (
                row.get('구분'), row.get('시간대'), row.get('이용금액'),
                row.get('방문인구(명)'), int(row.get('영역 구분'))
            ))
    
    def _load_inflow_data(self, cursor, df):
        """유입 데이터 로드"""
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO inflow_stats 
                (region_id, inflow_region_code, visitors, inflow_type, 
                 region_code, region_name, province_name)
                SELECT r.id, %s, %s, %s, %s, %s, %s
                FROM regions r WHERE r.region_code = %s
            """, (
                row.get('INFLOW_SGG_CD'), row.get('tot'), row.get('관내/관외'),
                row.get('SGG_CD'), row.get('SGG_NM'), row.get('SIDO_NM'),
                int(row.get('영역 구분'))
            ))
    
    def setup_complete_database(self):
        """완전한 데이터베이스 설정"""
        print("LangChain 기반 보고서 자동화 시스템 데이터베이스 설정을 시작합니다...")
        self.create_database_schema()
        print("데이터베이스 설정이 완료되었습니다!")

if __name__ == "__main__":
    manager = LangChainDatabaseManager()
    manager.setup_complete_database()

