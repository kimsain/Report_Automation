# Power BI 보고서 자동화 및 RAG 기반 인사이트 생성 시스템

이 프로젝트는 LangChain 프레임워크를 활용하여 구축된 보고서 자동화 및 RAG(Retrieval-Augmented Generation) 기반 인사이트 생성 시스템입니다.

## 프로젝트 발전 과정

### 1단계: MVP 개발 (LangChain 기반)
- 임베딩: LangChain `OpenAIEmbeddings` 활용
- 벡터 DB: LangChain `PGVector` 통합
- LLM: LangChain `ChatOpenAI` 체인 활용
- 텍스트 처리: LangChain `RecursiveCharacterTextSplitter`
- RAG: LangChain `RetrievalQA` 체인 구현

### 2단계: 최적화 (직접 구현)
- 임베딩: `openai.Embedding.create()` 직접 호출
- 벡터 DB: `psycopg2`로 PostgreSQL 직접 관리
- LLM: OpenAI API 직접 호출
- 텍스트 처리: 커스텀 분할 로직

## 주요 기능

### LangChain 컴포넌트
- **ChatOpenAI**: GPT-4o 기반 질의응답 및 인사이트 생성
- **OpenAIEmbeddings**: text-embedding-3-small 모델 활용
- **PGVector**: PostgreSQL 기반 벡터 데이터베이스
- **RetrievalQA**: RAG 기반 질의응답 체인
- **RecursiveCharacterTextSplitter**: 지능형 텍스트 분할

### 핵심 기능
1. **RAG 기반 질의응답**: 문서 검색 기반 자연어 질의응답
2. **자동 인사이트 생성**: 데이터 기반 인사이트 자동 생성 및 평가
3. **시멘틱 서치**: 의미 기반 유사 문서 검색
4. **벡터 데이터베이스 관리**: 실시간 문서 추가 및 업데이트
5. **성능 평가**: 시스템 품질 지표 자동 측정

## 파일 구조

```
langchain_mvp/
├── langchain_main.py                   # 통합 실행 스크립트
├── langchain_app.py                    # Flask 웹 애플리케이션
├── langchain_rag_system.py             # RetrievalQA 기반 RAG 시스템
├── langchain_insight_generator.py      # ChatOpenAI 기반 인사이트 생성
├── langchain_vector_db_manager.py      # PGVector 벡터 DB 관리
├── langchain_embedding_utils.py        # OpenAIEmbeddings 유틸리티
├── langchain_database_manager.py       # PostgreSQL 데이터베이스 관리
├── langchain_text_splitter.py          # RecursiveCharacterTextSplitter
└── README.md                           # 프로젝트 문서
```

### 기존 시스템 (분실된 파일들)
```
report_automation/
├── app.py                              # Flask 메인 애플리케이션
├── data_processor.py                   # 데이터 처리 모듈
├── embedding_generator.py              # 임베딩 생성 모듈
├── insight_generator.py                # 인사이트 생성 모듈
├── rag_system_v3.py                    # RAG 시스템
├── vector_db_manager.py                # 벡터 DB 관리
└── config.py                           # 설정 파일
```

### 주요 차이점

| 구성 요소 | 기존 시스템 | LangChain MVP | 주요 차이점 |
|-----------|-------------|---------------|-------------|
| **메인 앱** | `app.py` | `langchain_app.py` | 직접 구현 vs LangChain 통합 |
| **RAG 시스템** | `rag_system_v3.py` | `langchain_rag_system.py` | 커스텀 구현 vs RetrievalQA 체인 |
| **임베딩** | `embedding_generator.py` | `langchain_embedding_utils.py` | 직접 API 호출 vs OpenAIEmbeddings |
| **벡터 DB** | `vector_db_manager.py` | `langchain_vector_db_manager.py` | psycopg2 직접 vs PGVector 통합 |
| **인사이트** | `insight_generator.py` | `langchain_insight_generator.py` | 직접 구현 vs ChatOpenAI 체인 |
| **텍스트 분할** | 커스텀 로직 | `langchain_text_splitter.py` | 수동 분할 vs RecursiveCharacterTextSplitter |
| **설정 관리** | `config.py` | `langchain_config.py` | 단순 설정 vs 중앙화된 관리 |

