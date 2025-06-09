"""
LangChain 기반 RAG 시스템

이 모듈은 LangChain의 RetrievalQA 체인을 활용하여 
문서 검색 기반 질의응답 시스템을 구현합니다.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from langchain_config import LangChainConfig
from langchain_vector_db_manager import LangChainVectorDBManager
from langchain_embedding_utils import LangChainEmbeddingUtils

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = LangChainConfig.OPENAI_API_KEY

class LangChainRAGSystem:
    """
    LangChain 기반 RAG (Retrieval-Augmented Generation) 시스템 클래스
    """
    
    def __init__(self):
        """RAG 시스템 초기화"""
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name=LangChainConfig.OPENAI_COMPLETION_MODEL,
            temperature=0,
            openai_api_key=LangChainConfig.OPENAI_API_KEY
        )
        
        # 벡터 DB 매니저 초기화
        self.vector_manager = LangChainVectorDBManager()
        
        # 임베딩 유틸리티 초기화
        self.embedding_utils = LangChainEmbeddingUtils()
        
        # 프롬프트 템플릿 설정
        self._setup_prompt_template()
        
        # RAG 체인 초기화
        self.qa_chain = None
        self._initialize_rag_chain()
    
    def _setup_prompt_template(self):
        """프롬프트 템플릿 설정"""
        template = """
        당신은 축제 및 지역 데이터 분석 전문가입니다. 주어진 정보를 바탕으로 사용자의 질문에 정확하게 답변해 주세요.
        
        질문에 관련된 정보가 없는 경우, "제공된 데이터에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.
        
        ### 관련 정보:
        {context}
        
        ### 질문:
        {question}
        
        ### 답변:
        """
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _initialize_rag_chain(self):
        """RAG 체인 초기화"""
        try:
            # 벡터스토어 초기화
            self.vector_manager.initialize_vectorstore()
            
            # 리트리버 생성
            retriever = self.vector_manager.get_retriever(
                search_kwargs={"k": LangChainConfig.SIMILARITY_SEARCH_K}
            )
            
            # RAG 체인 생성
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True
            )
            
            print("RAG 시스템이 성공적으로 초기화되었습니다.")
            
        except Exception as e:
            print(f"RAG 시스템 초기화 중 오류 발생: {e}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        질의응답 수행
        
        Args:
            question (str): 사용자 질문
        
        Returns:
            Dict[str, Any]: 답변과 출처 문서
        """
        if not self.qa_chain:
            return {
                "answer": "RAG 시스템이 초기화되지 않았습니다.",
                "sources": [],
                "error": "System not initialized"
            }
        
        try:
            # RAG 체인 실행
            result = self.qa_chain({"query": question})
            
            # 응답 형식화
            response = {
                "answer": result["result"],
                "sources": [doc.page_content for doc in result["source_documents"]],
                "source_metadata": [doc.metadata for doc in result["source_documents"]]
            }
            
            return response
            
        except Exception as e:
            return {
                "answer": f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def similarity_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        유사도 검색 수행
        
        Args:
            query (str): 검색 쿼리
            k (int): 반환할 문서 수
        
        Returns:
            List[Dict[str, Any]]: 유사한 문서 리스트
        """
        k = k or LangChainConfig.SIMILARITY_SEARCH_K
        
        try:
            # 벡터 DB를 통한 유사도 검색
            results = self.vector_manager.similarity_search_with_score(query, k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"유사도 검색 중 오류 발생: {e}")
            return []
    
    def add_documents_to_knowledge_base(self, texts: List[str], metadatas: List[Dict] = None):
        """
        지식 베이스에 문서 추가
        
        Args:
            texts (List[str]): 추가할 텍스트 리스트
            metadatas (List[Dict]): 메타데이터 리스트
        """
        try:
            self.vector_manager.add_documents_from_texts(texts, metadatas)
            print(f"{len(texts)}개 문서가 지식 베이스에 추가되었습니다.")
        except Exception as e:
            print(f"문서 추가 중 오류 발생: {e}")
    
    def refresh_knowledge_base(self):
        """지식 베이스 새로고침 (데이터베이스에서 재로드)"""
        try:
            # 기존 컬렉션 삭제
            self.vector_manager.delete_collection()
            
            # 데이터베이스에서 문서 재로드
            self.vector_manager.add_documents_from_database()
            
            # RAG 체인 재초기화
            self._initialize_rag_chain()
            
            print("지식 베이스가 성공적으로 새로고침되었습니다.")
            
        except Exception as e:
            print(f"지식 베이스 새로고침 중 오류 발생: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 정보 반환"""
        try:
            # 벡터 DB 통계
            vector_stats = self.vector_manager.get_collection_stats()
            
            # 임베딩 통계
            embedding_stats = self.embedding_utils.get_embedding_statistics()
            
            return {
                "rag_chain_initialized": self.qa_chain is not None,
                "vector_db_stats": vector_stats,
                "embedding_stats": embedding_stats,
                "llm_model": LangChainConfig.OPENAI_COMPLETION_MODEL,
                "embedding_model": LangChainConfig.OPENAI_EMBEDDING_MODEL
            }
            
        except Exception as e:
            return {
                "error": f"상태 정보 조회 중 오류 발생: {e}"
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        배치 질의응답 수행
        
        Args:
            questions (List[str]): 질문 리스트
        
        Returns:
            List[Dict[str, Any]]: 답변 리스트
        """
        results = []
        for question in questions:
            result = self.query(question)
            result["question"] = question
            results.append(result)
        
        return results

if __name__ == "__main__":
    # 테스트 코드
    rag_system = LangChainRAGSystem()
    
    # 시스템 상태 확인
    status = rag_system.get_system_status()
    print("시스템 상태:", status)
    
    # 테스트 질문
    test_questions = [
        "축제 기간 동안 방문자 수는 얼마나 되나요?",
        "주요 매출 업종은 무엇인가요?",
        "시간대별 방문 패턴은 어떻게 되나요?"
    ]
    
    for question in test_questions:
        result = rag_system.query(question)
        print(f"\n질문: {question}")
        print(f"답변: {result['answer']}")
        print(f"출처 수: {len(result['sources'])}")

