import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 테스트 쿼리 목록
test_queries = [
    "축제 기간 동안 방문객 수는 얼마였나요?",
    "주요 매출 업종은 무엇인가요?",
    "시간대별 방문객 수는 어떻게 되나요?",
    "축제 전과 비교해서 매출 증가율은 얼마인가요?",
    "어느 지역에서 방문객이 가장 많이 왔나요?",
    "남성과 여성 방문객 비율은 어떻게 되나요?",
    "60대 방문객 수는 얼마인가요?",
    "홍삼제품 매출액은 얼마인가요?",
    "14시에서 17시 사이의 매출액은 얼마인가요?",
    "금산군 내부에서 온 방문객 수는 얼마인가요?"
]

def test_vector_store():
    """벡터 저장소 로드 및 테스트"""
    try:
        # 벡터 저장소가 존재하는지 확인
        if not os.path.exists("faiss_index"):
            print("벡터 저장소가 존재하지 않습니다. 먼저 rag_system.py를 실행하세요.")
            return False
        
        # 임베딩 초기화 (API 키 필요)
        embeddings = OpenAIEmbeddings()
        
        # 벡터 저장소 로드 (안전하지 않은 역직렬화 허용)
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # 간단한 쿼리로 테스트
        test_query = "축제 기간 방문객"
        results = vector_store.similarity_search(test_query, k=1)
        
        if results and len(results) > 0:
            print("벡터 저장소 테스트 성공!")
            print(f"테스트 쿼리: '{test_query}'")
            print(f"검색 결과: {results[0].page_content[:100]}...")
            return True
        else:
            print("벡터 저장소 테스트 실패: 검색 결과가 없습니다.")
            return False
    
    except Exception as e:
        print(f"벡터 저장소 테스트 중 오류 발생: {e}")
        return False

def test_api_key():
    """OpenAI API 키 테스트"""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key or api_key == "your-api-key-here":
        print("OpenAI API 키가 설정되지 않았습니다.")
        print("rag_system.py와 app.py 파일에서 API 키를 설정하세요.")
        return False
    
    print("OpenAI API 키가 설정되어 있습니다.")
    return True

def run_tests():
    """모든 테스트 실행"""
    print("=== RAG 시스템 테스트 시작 ===")
    
    # API 키 테스트
    api_key_result = test_api_key()
    
    if not api_key_result:
        print("API 키 설정이 필요합니다. 테스트를 중단합니다.")
        return
    
    # 벡터 저장소 테스트
    vector_store_result = test_vector_store()
    
    if not vector_store_result:
        print("벡터 저장소 테스트에 실패했습니다. 테스트를 중단합니다.")
        return
    
    print("\n=== 테스트 쿼리 목록 ===")
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. {query}")
    
    print("\n시스템 테스트가 완료되었습니다.")
    print("웹 인터페이스를 통해 위의 테스트 쿼리를 실행해보세요.")
    print("웹 서버 실행 방법: python app.py")
    print("웹 브라우저에서 http://localhost:5000 접속")

if __name__ == "__main__":
    # 환경 변수 로드
    load_dotenv()
    
    # 테스트 실행
    run_tests()
