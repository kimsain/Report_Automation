"""
LangChain 기반 임베딩 유틸리티

이 모듈은 LangChain의 OpenAIEmbeddings를 활용하여 텍스트 임베딩을 생성하고 관리하는 기능을 제공합니다.
"""

import os
import json
import psycopg2
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_config import LangChainConfig

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = LangChainConfig.OPENAI_API_KEY

class LangChainEmbeddingUtils:
    """
    LangChain 기반 임베딩 유틸리티 클래스
    """
    
    def __init__(self):
        """임베딩 유틸리티 초기화"""
        self.db_params = LangChainConfig.get_db_connection_params()
        self.embeddings = OpenAIEmbeddings(
            model=LangChainConfig.OPENAI_EMBEDDING_MODEL,
            openai_api_key=LangChainConfig.OPENAI_API_KEY
        )
    
    def connect_to_db(self):
        """데이터베이스 연결"""
        return psycopg2.connect(**self.db_params)
    
    def generate_embeddings_from_database(self):
        """데이터베이스에서 데이터를 추출하여 임베딩 생성"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        # 대상 테이블 정의 및 데이터 추출 쿼리
        tables = {
            "festival_stats": """
                SELECT r.region_code, f.period_type, f.visitors, f.sales_amount, 
                       f.main_age_group, f.main_business_type 
                FROM festival_stats f 
                JOIN regions r ON f.region_id = r.id
            """,
            "population_stats": """
                SELECT r.region_code, p.gender, p.age_group, p.period_type, p.visitors 
                FROM population_stats p 
                JOIN regions r ON p.region_id = r.id
            """,
            "sales_stats": """
                SELECT r.region_code, s.business_type, s.sales_amount 
                FROM sales_stats s 
                JOIN regions r ON s.region_id = r.id
            """,
            "hourly_stats": """
                SELECT r.region_code, h.period_type, h.time_period, h.visitors, h.sales_amount 
                FROM hourly_stats h 
                JOIN regions r ON h.region_id = r.id
            """,
            "inflow_stats": """
                SELECT r.region_code, i.inflow_region_code, i.visitors, i.inflow_type, 
                       i.region_name, i.province_name 
                FROM inflow_stats i 
                JOIN regions r ON i.region_id = r.id
            """
        }
        
        try:
            # 기존 임베딩 데이터 삭제
            cursor.execute("DELETE FROM embedding_store")
            conn.commit()
            
            # 각 테이블에서 데이터 추출 및 임베딩 저장
            for table, query in tables.items():
                print(f"\n📂 {table} 테이블에서 문장 생성 및 임베딩 중...")
                cursor.execute(query)
                rows = cursor.fetchall()
                
                documents = []
                for row in tqdm(rows, desc=f"Processing {table}"):
                    # 문장 생성 로직
                    text, metadata = self._generate_text_and_metadata(table, row)
                    if text:
                        documents.append(Document(page_content=text, metadata=metadata))
                
                # 배치로 임베딩 생성 및 저장
                if documents:
                    self._save_embeddings_batch(cursor, table, documents)
                    conn.commit()
                    print(f"✅ {table} 테이블 완료. 저장된 문장 수: {len(documents)}")
            
            print("\n🎉 모든 테이블 임베딩 완료")
            
        except Exception as e:
            print(f"임베딩 생성 중 오류 발생: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def _generate_text_and_metadata(self, table: str, row: tuple) -> tuple:
        """테이블별 문장 생성 및 메타데이터 추출"""
        if table == "festival_stats":
            text = f"영역 {row[0]}의 {row[1]} 기간: 방문자 수 {row[2]}명, 매출 {row[3]}억, 주요 연령층 {row[4]}, 주 매출 업종: {row[5]}"
            metadata = {"region_code": row[0], "period_type": row[1], "table": table}
            
        elif table == "population_stats":
            text = f"{row[0]} 지역 {row[1]} {row[2]}의 {row[3]} 방문객 수는 {row[4]}명입니다."
            metadata = {"region_code": row[0], "gender": row[1], "age_group": row[2], "period_type": row[3], "table": table}
            
        elif table == "sales_stats":
            text = f"{row[0]} 지역의 {row[1]} 업종 매출은 {row[2]}입니다."
            metadata = {"region_code": row[0], "business_type": row[1], "table": table}
            
        elif table == "hourly_stats":
            text = f"{row[0]} 지역 {row[1]} 시간대({row[2]}) 방문객 수는 {row[3]}명이고 매출은 {row[4]}입니다."
            metadata = {"region_code": row[0], "period_type": row[1], "time_period": row[2], "table": table}
            
        elif table == "inflow_stats":
            text = f"{row[5]} {row[4]}에서 유입된 인구는 {row[2]}명이며 유입유형은 {row[3]}입니다."
            metadata = {"region_code": row[0], "inflow_type": row[3], "region_name": row[4], "table": table}
            
        else:
            return None, None
        
        return text, metadata
    
    def _save_embeddings_batch(self, cursor, table: str, documents: List[Document]):
        """배치로 임베딩 생성 및 저장"""
        # 텍스트 리스트 추출
        texts = [doc.page_content for doc in documents]
        
        # LangChain을 사용하여 임베딩 생성
        embeddings = self.embeddings.embed_documents(texts)
        
        # 데이터베이스에 저장
        for doc, embedding in zip(documents, embeddings):
            cursor.execute("""
                INSERT INTO embedding_store (table_name, original_text, metadata, embedding)
                VALUES (%s, %s, %s, %s)
            """, (table, doc.page_content, json.dumps(doc.metadata), embedding))
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """유사한 문서 검색"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            
            # 유사도 검색
            cursor.execute("""
                SELECT original_text, metadata, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM embedding_store
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, k))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "text": row[0],
                    "metadata": row[1],
                    "similarity": float(row[2])
                })
            
            return results
            
        except Exception as e:
            print(f"유사 문서 검색 중 오류 발생: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """임베딩 통계 정보 반환"""
        conn = self.connect_to_db()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT table_name, COUNT(*) as count
                FROM embedding_store
                GROUP BY table_name
                ORDER BY count DESC
            """)
            
            table_stats = {}
            total_count = 0
            for row in cursor.fetchall():
                table_stats[row[0]] = row[1]
                total_count += row[1]
            
            return {
                "total_embeddings": total_count,
                "table_statistics": table_stats,
                "embedding_model": LangChainConfig.OPENAI_EMBEDDING_MODEL,
                "vector_dimension": LangChainConfig.VECTOR_DIMENSION
            }
            
        except Exception as e:
            print(f"통계 정보 조회 중 오류 발생: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    utils = LangChainEmbeddingUtils()
    utils.generate_embeddings_from_database()

