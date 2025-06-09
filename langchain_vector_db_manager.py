"""
LangChain 기반 벡터 데이터베이스 관리 클래스

이 모듈은 LangChain의 PGVector를 활용하여 벡터 데이터베이스를 관리하는 기능을 제공합니다.
"""

import os
import psycopg2
from typing import List, Dict, Any, Optional
from langchain.vectorstores import PGVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_config import LangChainConfig

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = LangChainConfig.OPENAI_API_KEY

class LangChainVectorDBManager:
    """
    LangChain PGVector를 활용한 벡터 데이터베이스 관리 클래스
    """
    
    def __init__(self, collection_name: str = None):
        """
        벡터 데이터베이스 매니저 초기화
        
        Args:
            collection_name (str): 컬렉션(테이블) 이름
        """
        self.collection_name = collection_name or LangChainConfig.COLLECTION_NAME
        
        # DB 연결 정보 구성
        self.connection_string = LangChainConfig.get_connection_string()
        
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model=LangChainConfig.OPENAI_EMBEDDING_MODEL,
            openai_api_key=LangChainConfig.OPENAI_API_KEY
        )
        
        # 텍스트 스플리터 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=LangChainConfig.CHUNK_SIZE,
            chunk_overlap=LangChainConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "]
        )
        
        # PGVector 인스턴스
        self.vectorstore = None
    
    def initialize_vectorstore(self):
        """벡터스토어 초기화"""
        try:
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            print(f"벡터스토어 '{self.collection_name}' 초기화 완료")
        except Exception as e:
            print(f"벡터스토어 초기화 중 오류 발생: {e}")
    
    def add_documents_from_texts(self, texts: List[str], metadatas: List[Dict] = None):
        """텍스트 리스트로부터 문서 추가"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            # 텍스트를 Document 객체로 변환
            documents = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                documents.append(Document(page_content=text, metadata=metadata))
            
            # 벡터스토어에 추가
            self.vectorstore.add_documents(documents)
            print(f"{len(documents)}개 문서가 벡터스토어에 추가되었습니다.")
            
        except Exception as e:
            print(f"문서 추가 중 오류 발생: {e}")
    
    def add_documents_from_database(self):
        """데이터베이스에서 문서를 추출하여 벡터스토어에 추가"""
        conn = psycopg2.connect(**LangChainConfig.get_db_connection_params())
        cursor = conn.cursor()
        
        try:
            # embedding_store 테이블에서 데이터 조회
            cursor.execute("""
                SELECT original_text, metadata, table_name
                FROM embedding_store
                ORDER BY id
            """)
            
            documents = []
            for row in cursor.fetchall():
                text, metadata, table_name = row
                if isinstance(metadata, dict):
                    metadata['source_table'] = table_name
                else:
                    metadata = {'source_table': table_name}
                
                documents.append(Document(page_content=text, metadata=metadata))
            
            if documents:
                if not self.vectorstore:
                    # 첫 번째 문서로 벡터스토어 생성
                    self.vectorstore = PGVector.from_documents(
                        documents=documents[:1],
                        embedding=self.embeddings,
                        connection_string=self.connection_string,
                        collection_name=self.collection_name
                    )
                    
                    # 나머지 문서들 추가
                    if len(documents) > 1:
                        self.vectorstore.add_documents(documents[1:])
                else:
                    self.vectorstore.add_documents(documents)
                
                print(f"{len(documents)}개 문서가 벡터스토어에 추가되었습니다.")
            
        except Exception as e:
            print(f"데이터베이스에서 문서 추가 중 오류 발생: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """유사도 검색"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        k = k or LangChainConfig.SIMILARITY_SEARCH_K
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"유사도 검색 중 오류 발생: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = None) -> List[tuple]:
        """점수와 함께 유사도 검색"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        k = k or LangChainConfig.SIMILARITY_SEARCH_K
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"점수와 함께 유사도 검색 중 오류 발생: {e}")
            return []
    
    def get_retriever(self, search_kwargs: Dict = None):
        """리트리버 반환"""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        search_kwargs = search_kwargs or {"k": LangChainConfig.SIMILARITY_SEARCH_K}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def delete_collection(self):
        """컬렉션 삭제"""
        try:
            conn = psycopg2.connect(**LangChainConfig.get_db_connection_params())
            cursor = conn.cursor()
            
            # 컬렉션 테이블 삭제
            cursor.execute(f"DROP TABLE IF EXISTS langchain_pg_collection CASCADE")
            cursor.execute(f"DROP TABLE IF EXISTS langchain_pg_embedding CASCADE")
            
            conn.commit()
            print(f"컬렉션 '{self.collection_name}'이 삭제되었습니다.")
            
        except Exception as e:
            print(f"컬렉션 삭제 중 오류 발생: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보 반환"""
        try:
            conn = psycopg2.connect(**LangChainConfig.get_db_connection_params())
            cursor = conn.cursor()
            
            # 컬렉션 문서 수 조회
            cursor.execute("""
                SELECT COUNT(*) 
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
            """, (self.collection_name,))
            
            document_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
            
            return {
                "collection_name": self.collection_name,
                "document_count": document_count,
                "embedding_model": LangChainConfig.OPENAI_EMBEDDING_MODEL,
                "vector_dimension": LangChainConfig.VECTOR_DIMENSION
            }
            
        except Exception as e:
            print(f"통계 정보 조회 중 오류 발생: {e}")
            return {}
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

if __name__ == "__main__":
    manager = LangChainVectorDBManager()
    manager.add_documents_from_database()

