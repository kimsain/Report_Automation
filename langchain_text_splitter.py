"""
LangChain 기반 텍스트 분할 모듈

이 모듈은 LangChain의 텍스트 스플리터를 활용하여 다양한 형태의 텍스트를 분할하는 기능을 제공합니다.
"""

from typing import List, Dict, Any, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)
from langchain.schema import Document
from langchain_config import LangChainConfig

class LangChainTextSplitter:
    """
    LangChain 기반 텍스트 분할 클래스
    """
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None,
                 separators: List[str] = None):
        """
        텍스트 스플리터 초기화
        
        Args:
            chunk_size (int): 청크 크기
            chunk_overlap (int): 청크 겹침 크기
            separators (List[str]): 분리자 리스트
        """
        self.chunk_size = chunk_size or LangChainConfig.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or LangChainConfig.CHUNK_OVERLAP
        self.separators = separators or ["\n\n", "\n", ".", " "]
        
        # 기본 스플리터 초기화
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )
        
        self.character_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator="\n"
        )
        
        self.token_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def split_text_recursive(self, text: str) -> List[str]:
        """재귀적 문자 분할"""
        return self.recursive_splitter.split_text(text)
    
    def split_text_by_character(self, text: str, separator: str = "\n") -> List[str]:
        """문자 기반 분할"""
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=separator
        )
        return splitter.split_text(text)
    
    def split_text_by_token(self, text: str) -> List[str]:
        """토큰 기반 분할"""
        return self.token_splitter.split_text(text)
    
    def split_documents_recursive(self, documents: List[Document]) -> List[Document]:
        """문서 리스트를 재귀적으로 분할"""
        return self.recursive_splitter.split_documents(documents)
    
    def split_documents_by_character(self, documents: List[Document], separator: str = "\n") -> List[Document]:
        """문서 리스트를 문자 기반으로 분할"""
        splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=separator
        )
        return splitter.split_documents(documents)
    
    def split_documents_by_token(self, documents: List[Document]) -> List[Document]:
        """문서 리스트를 토큰 기반으로 분할"""
        return self.token_splitter.split_documents(documents)
    
    def create_documents_from_texts(self, 
                                   texts: List[str], 
                                   metadatas: List[Dict] = None,
                                   split_method: str = "recursive") -> List[Document]:
        """
        텍스트 리스트로부터 분할된 문서 생성
        
        Args:
            texts (List[str]): 텍스트 리스트
            metadatas (List[Dict]): 메타데이터 리스트
            split_method (str): 분할 방법 ("recursive", "character", "token")
        
        Returns:
            List[Document]: 분할된 문서 리스트
        """
        # 문서 생성
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            documents.append(Document(page_content=text, metadata=metadata))
        
        # 분할 방법에 따라 처리
        if split_method == "recursive":
            return self.split_documents_recursive(documents)
        elif split_method == "character":
            return self.split_documents_by_character(documents)
        elif split_method == "token":
            return self.split_documents_by_token(documents)
        else:
            raise ValueError(f"지원하지 않는 분할 방법: {split_method}")
    
    def split_database_content(self, table_data: Dict[str, List[str]]) -> Dict[str, List[Document]]:
        """
        데이터베이스 테이블 내용을 분할
        
        Args:
            table_data (Dict[str, List[str]]): 테이블별 텍스트 데이터
        
        Returns:
            Dict[str, List[Document]]: 테이블별 분할된 문서
        """
        split_results = {}
        
        for table_name, texts in table_data.items():
            documents = []
            for i, text in enumerate(texts):
                metadata = {
                    "source_table": table_name,
                    "chunk_index": i,
                    "original_length": len(text)
                }
                documents.append(Document(page_content=text, metadata=metadata))
            
            # 재귀적 분할 적용
            split_documents = self.split_documents_recursive(documents)
            
            # 분할 후 메타데이터 업데이트
            for j, doc in enumerate(split_documents):
                doc.metadata.update({
                    "split_index": j,
                    "split_length": len(doc.page_content)
                })
            
            split_results[table_name] = split_documents
        
        return split_results
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """텍스트 통계 정보 반환"""
        chunks_recursive = self.split_text_recursive(text)
        chunks_character = self.split_text_by_character(text)
        chunks_token = self.split_text_by_token(text)
        
        return {
            "original_length": len(text),
            "recursive_chunks": len(chunks_recursive),
            "character_chunks": len(chunks_character),
            "token_chunks": len(chunks_token),
            "avg_chunk_size_recursive": sum(len(chunk) for chunk in chunks_recursive) / len(chunks_recursive) if chunks_recursive else 0,
            "avg_chunk_size_character": sum(len(chunk) for chunk in chunks_character) / len(chunks_character) if chunks_character else 0,
            "avg_chunk_size_token": sum(len(chunk) for chunk in chunks_token) / len(chunks_token) if chunks_token else 0,
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap
        }
    
    def optimize_chunk_size(self, texts: List[str], target_chunks: int = 10) -> int:
        """
        목표 청크 수에 맞는 최적 청크 크기 계산
        
        Args:
            texts (List[str]): 분석할 텍스트 리스트
            target_chunks (int): 목표 청크 수
        
        Returns:
            int: 최적 청크 크기
        """
        total_length = sum(len(text) for text in texts)
        estimated_chunk_size = total_length // target_chunks
        
        # 최소/최대 청크 크기 제한
        min_chunk_size = 100
        max_chunk_size = 2000
        
        optimized_size = max(min_chunk_size, min(estimated_chunk_size, max_chunk_size))
        
        return optimized_size

if __name__ == "__main__":
    # 테스트 코드
    splitter = LangChainTextSplitter()
    
    test_text = """
    이것은 테스트 텍스트입니다. 
    여러 문단으로 구성되어 있습니다.
    
    두 번째 문단입니다.
    LangChain 텍스트 스플리터의 기능을 테스트합니다.
    
    세 번째 문단입니다.
    다양한 분할 방법을 시험해볼 수 있습니다.
    """
    
    stats = splitter.get_text_statistics(test_text)
    print("텍스트 통계:", stats)

