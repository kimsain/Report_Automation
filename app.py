import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# 환경 변수 로드
load_dotenv()

# Flask 앱 초기화
app = Flask(__name__)

# OpenAI API 키 설정 (실제 사용 시 .env 파일에서 로드하거나 사용자 입력으로 받아야 함)
# OpenAI API 키는 .env 파일에서 로드됨
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 벡터 저장소 로드
def load_vector_store():
    """저장된 벡터 저장소 로드"""
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"벡터 저장소 로드 중 오류 발생: {e}")
        return None

# RAG 시스템 생성
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
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # RAG 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa_chain

# 전역 변수로 RAG 시스템 초기화
vector_store = None
qa_chain = None

# 최신 Flask 버전에서는 before_first_request가 제거됨
# 대신 앱 시작 시 초기화 함수를 직접 호출
def initialize():
    global vector_store, qa_chain
    vector_store = load_vector_store()
    if vector_store:
        qa_chain = create_rag_system(vector_store)
    else:
        print("벡터 저장소를 로드할 수 없습니다. RAG 시스템을 초기화할 수 없습니다.")

# 홈 페이지 라우트
@app.route('/')
def home():
    return render_template('index.html')

# 질문 처리 API 엔드포인트
@app.route('/api/query', methods=['POST'])
def query():
    global qa_chain
    
    # 요청에서 질문 추출
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': '질문이 제공되지 않았습니다.'}), 400
    
    if not qa_chain:
        return jsonify({'error': 'RAG 시스템이 초기화되지 않았습니다.'}), 500
    
    try:
        # RAG 시스템을 사용하여 질문에 답변
        result = qa_chain({"query": question})
        
        # 응답 형식화
        response = {
            'answer': result['result'],
            'sources': [doc.page_content for doc in result['source_documents']]
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'질문 처리 중 오류 발생: {str(e)}'}), 500

# 메인 실행
if __name__ == '__main__':
    # 템플릿 디렉토리 생성
    os.makedirs('templates', exist_ok=True)
    
    # 벡터 저장소 로드 및 RAG 시스템 초기화
    vector_store = load_vector_store()
    if vector_store:
        qa_chain = create_rag_system(vector_store)
    else:
        print("벡터 저장소를 로드할 수 없습니다. RAG 시스템을 초기화할 수 없습니다.")
    
    # 앱 실행
    app.run(host='0.0.0.0', port=5000, debug=True)
