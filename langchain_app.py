"""
LangChain 기반 Flask 애플리케이션

이 모듈은 LangChain을 활용한 보고서 자동화 시스템의 웹 API를 제공합니다.
"""

import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_config import LangChainConfig
from langchain_rag_system import LangChainRAGSystem
from langchain_insight_generator import LangChainInsightGenerator
from langchain_database_manager import LangChainDatabaseManager
from langchain_embedding_utils import LangChainEmbeddingUtils

# 환경 변수 로드
load_dotenv()

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 설정

# LangChain 컴포넌트 초기화
rag_system = None
insight_generator = None
db_manager = None
embedding_utils = None

def initialize_components():
    """LangChain 컴포넌트 초기화"""
    global rag_system, insight_generator, db_manager, embedding_utils
    
    try:
        print("LangChain 컴포넌트를 초기화하는 중...")
        
        # 데이터베이스 매니저 초기화
        db_manager = LangChainDatabaseManager()
        
        # 임베딩 유틸리티 초기화
        embedding_utils = LangChainEmbeddingUtils()
        
        # RAG 시스템 초기화
        rag_system = LangChainRAGSystem()
        
        # 인사이트 생성기 초기화
        insight_generator = LangChainInsightGenerator()
        
        print("LangChain 컴포넌트 초기화 완료!")
        
    except Exception as e:
        print(f"컴포넌트 초기화 중 오류 발생: {e}")

# HTML 템플릿
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LangChain 보고서 자동화 시스템</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
        .input-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, button { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #007bff; color: white; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .source { background: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <h1>🤖 LangChain 보고서 자동화 시스템</h1>
    
    <div class="container">
        <h2>📊 RAG 기반 질의응답</h2>
        <div class="input-group">
            <label for="question">질문을 입력하세요:</label>
            <textarea id="question" rows="3" placeholder="예: 축제 기간 동안 방문자 수는 얼마나 되나요?"></textarea>
        </div>
        <button onclick="askQuestion()">질문하기</button>
        <div id="answer-result"></div>
    </div>
    
    <div class="container">
        <h2>🔍 유사도 검색</h2>
        <div class="input-group">
            <label for="search-query">검색어를 입력하세요:</label>
            <input type="text" id="search-query" placeholder="예: 매출 증가">
        </div>
        <button onclick="searchSimilar()">검색하기</button>
        <div id="search-result"></div>
    </div>
    
    <div class="container">
        <h2>⚙️ 시스템 관리</h2>
        <button onclick="getSystemStatus()">시스템 상태 확인</button>
        <button onclick="refreshKnowledgeBase()">지식 베이스 새로고침</button>
        <div id="system-result"></div>
    </div>

    <script>
        async function askQuestion() {
            const question = document.getElementById('question').value;
            if (!question.trim()) {
                alert('질문을 입력해주세요.');
                return;
            }
            
            const resultDiv = document.getElementById('answer-result');
            resultDiv.innerHTML = '<p>답변을 생성하는 중...</p>';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="result error">오류: ${data.error}</div>`;
                } else {
                    let html = `<div class="result">
                        <h3>답변:</h3>
                        <p>${data.answer}</p>
                        <h4>참고 자료:</h4>`;
                    
                    data.sources.forEach((source, index) => {
                        html += `<div class="source">출처 ${index + 1}: ${source}</div>`;
                    });
                    
                    html += '</div>';
                    resultDiv.innerHTML = html;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">요청 중 오류가 발생했습니다: ${error.message}</div>`;
            }
        }
        
        async function searchSimilar() {
            const query = document.getElementById('search-query').value;
            if (!query.trim()) {
                alert('검색어를 입력해주세요.');
                return;
            }
            
            const resultDiv = document.getElementById('search-result');
            resultDiv.innerHTML = '<p>검색하는 중...</p>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, k: 5 })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = `<div class="result error">오류: ${data.error}</div>`;
                } else {
                    let html = '<div class="result"><h3>검색 결과:</h3>';
                    
                    data.results.forEach((result, index) => {
                        html += `<div class="source">
                            <strong>결과 ${index + 1} (유사도: ${result.similarity_score.toFixed(4)})</strong><br>
                            ${result.content}
                        </div>`;
                    });
                    
                    html += '</div>';
                    resultDiv.innerHTML = html;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">검색 중 오류가 발생했습니다: ${error.message}</div>`;
            }
        }
        
        async function getSystemStatus() {
            const resultDiv = document.getElementById('system-result');
            resultDiv.innerHTML = '<p>상태를 확인하는 중...</p>';
            
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                let html = '<div class="result"><h3>시스템 상태:</h3>';
                html += `<p><strong>RAG 시스템:</strong> ${data.rag_chain_initialized ? '✅ 정상' : '❌ 오류'}</p>`;
                html += `<p><strong>LLM 모델:</strong> ${data.llm_model}</p>`;
                html += `<p><strong>임베딩 모델:</strong> ${data.embedding_model}</p>`;
                
                if (data.vector_db_stats) {
                    html += `<p><strong>벡터 DB 문서 수:</strong> ${data.vector_db_stats.document_count}</p>`;
                }
                
                if (data.embedding_stats) {
                    html += `<p><strong>총 임베딩 수:</strong> ${data.embedding_stats.total_embeddings}</p>`;
                }
                
                html += '</div>';
                resultDiv.innerHTML = html;
                
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">상태 확인 중 오류가 발생했습니다: ${error.message}</div>`;
            }
        }
        
        async function refreshKnowledgeBase() {
            const resultDiv = document.getElementById('system-result');
            resultDiv.innerHTML = '<p>지식 베이스를 새로고침하는 중...</p>';
            
            try {
                const response = await fetch('/api/refresh', { method: 'POST' });
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.innerHTML = '<div class="result success">지식 베이스가 성공적으로 새로고침되었습니다.</div>';
                } else {
                    resultDiv.innerHTML = `<div class="result error">오류: ${data.error}</div>`;
                }
                
            } catch (error) {
                resultDiv.innerHTML = `<div class="result error">새로고침 중 오류가 발생했습니다: ${error.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """홈 페이지"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/query', methods=['POST'])
def query():
    """RAG 기반 질의응답 API"""
    global rag_system
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': '질문이 제공되지 않았습니다.'}), 400
        
        if not rag_system:
            return jsonify({'error': 'RAG 시스템이 초기화되지 않았습니다.'}), 500
        
        # RAG 시스템을 사용하여 질문에 답변
        result = rag_system.query(question)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'질문 처리 중 오류 발생: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def search():
    """유사도 검색 API"""
    global rag_system
    
    try:
        data = request.json
        query = data.get('query', '')
        k = data.get('k', 5)
        
        if not query:
            return jsonify({'error': '검색어가 제공되지 않았습니다.'}), 400
        
        if not rag_system:
            return jsonify({'error': 'RAG 시스템이 초기화되지 않았습니다.'}), 500
        
        # 유사도 검색 수행
        results = rag_system.similarity_search(query, k)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': f'검색 중 오류 발생: {str(e)}'}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """시스템 상태 확인 API"""
    global rag_system
    
    try:
        if not rag_system:
            return jsonify({'error': 'RAG 시스템이 초기화되지 않았습니다.'}), 500
        
        status_info = rag_system.get_system_status()
        return jsonify(status_info)
        
    except Exception as e:
        return jsonify({'error': f'상태 확인 중 오류 발생: {str(e)}'}), 500

@app.route('/api/refresh', methods=['POST'])
def refresh():
    """지식 베이스 새로고침 API"""
    global rag_system
    
    try:
        if not rag_system:
            return jsonify({'error': 'RAG 시스템이 초기화되지 않았습니다.'}), 500
        
        rag_system.refresh_knowledge_base()
        return jsonify({'success': True, 'message': '지식 베이스가 성공적으로 새로고침되었습니다.'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'새로고침 중 오류 발생: {str(e)}'}), 500

@app.route('/api/insights', methods=['POST'])
def generate_insights():
    """인사이트 생성 API"""
    global insight_generator
    
    try:
        data = request.json
        crym = data.get('crym')
        sido = data.get('sido')
        sigungu = data.get('sigungu')
        subject = data.get('subject')
        statements = data.get('statements', [])
        
        if not all([crym, sido, sigungu, subject]):
            return jsonify({'error': '필수 파라미터가 누락되었습니다.'}), 400
        
        if not insight_generator:
            return jsonify({'error': '인사이트 생성기가 초기화되지 않았습니다.'}), 500
        
        # 인사이트 생성
        result_df = insight_generator.generate_insights_from_data(
            crym, sido, sigungu, subject, statements
        )
        
        # DataFrame을 JSON으로 변환
        insights = result_df.to_dict('records')
        
        return jsonify({'insights': insights})
        
    except Exception as e:
        return jsonify({'error': f'인사이트 생성 중 오류 발생: {str(e)}'}), 500

if __name__ == '__main__':
    # 컴포넌트 초기화
    initialize_components()
    
    # Flask 앱 실행
    app.run(host='0.0.0.0', port=5000, debug=True)

