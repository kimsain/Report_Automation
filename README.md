# Power BI 보고서 자동화 및 RAG 기반 인사이트 생성 시스템

## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [주요 기능](#주요-기능)  
3. [디렉터리 구조](#디렉터리-구조)  
4. [설치 및 환경 설정](#설치-및-환경-설정)  
5. [사용 방법](#사용-방법)  
6. [스크립트 설명](#스크립트-설명)  
7. [라이선스](#라이선스)  

---

## 프로젝트 개요
이 프로젝트는  
- Power BI 대시보드용 데이터를 전처리(`data_processor.py`)  
- OpenAI 임베딩을 생성(`embedding_generator.py`)  
- RAG(Retrieval-Augmented Generation) 기반 QA API를 제공(`app.py`)  
- 자동으로 인사이트를 생성(`insight_generator.py`)  
하는 일련의 파이프라인을 Flask 웹 애플리케이션 형태로 구현한 시스템입니다.

CSV 형태의 다양한 입력 데이터를 바탕으로 Vector Store(FAISS)를 구성하고, OpenAI와 LangChain을 활용하여 사용자 질의에 대한 답변 및 근거 소스를 제공합니다.

---

## 주요 기능
- **데이터 전처리**: 원본 CSV를 읽어 Power BI용 히스토그램, 지도 입력 데이터 등으로 가공  
- **임베딩 생성**: OpenAI Embeddings API로 문서별 벡터 생성 및 FAISS 인덱싱  
- **RAG QA 서비스**: Flask + LangChain으로 구현한 질의응답 API (`/api/qa`)  
- **인사이트 자동 생성**: 분석 결과를 요약·추출하여 텍스트 인사이트 생성  
- **테스트 스크립트**: 각 단계별 기능 검증용 테스트 (`test_system.py`)

---

## 디렉터리 구조
```
.
├── app.py                       # Flask 웹 애플리케이션 진입점
├── data/                        # 원본 및 중간 가공용 CSV 파일
│   ├── data_state.csv
│   ├── histo_for_powerbi.csv
│   └── …  
├── data_processor.py           # CSV 전처리 모듈
├── embedding_generator.py      # 문서 임베딩 생성 스크립트
├── embedding_store/            # FAISS 인덱스 및 메타데이터 (자동 생성)
├── insight_generator.py        # 인사이트 자동 생성 모듈
├── inspect_similar_docs.py     # 유사 문서 검토용 스크립트
├── rag_system_v1.py            # RAG 버전별 구현 예제
├── rag_system_v2.py
├── rag_system_v3.py
├── test_system.py              # 전체 파이프라인 테스트
└── .env                         # 환경 변수 (OpenAI API 키 등)
```

---

## 설치 및 환경 설정

1. **Python 3.8+** 설치  
2. 가상환경 생성 및 활성화  
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scriptsctivate
   ```
3. 필수 패키지 설치  
   ```bash
   pip install flask python-dotenv pandas langchain openai faiss-cpu langchain-community
   ```
4. `.env` 파일 생성  
   ```dotenv
   OPENAI_API_KEY=your_openai_api_key_here
   ```
5. FAISS 인덱스를 저장할 디렉터리 생성 (최초 실행 시 자동 생성됨)  
   ```bash
   mkdir embedding_store
   ```

---

## 사용 방법

### 1) 데이터 전처리
```bash
python data_processor.py   --input-dir ./data   --output-dir ./data/processed
```

### 2) 임베딩 및 벡터 스토어 생성
```bash
python embedding_generator.py   --data-dir ./data/processed   --store-dir ./embedding_store
```

### 3) Flask 서버 실행
```bash
python app.py
```
- 기본 포트: `http://localhost:5000`
- **엔드포인트**  
  - `GET  /` : 간단한 웹 UI (템플릿 폴더 활용)  
  - `POST /api/qa` : JSON `{ "question": "질문내용" }` 형태로 질의  
    ```json
    {
      "answer": "답변 텍스트",
      "sources": ["출처 문서 내용1", "출처 문서 내용2", ...]
    }
    ```

### 4) 인사이트 자동 생성 (선택)
```bash
python insight_generator.py   --data-dir ./data/processed   --output ./insights.txt
```

---

## 스크립트 설명

- **data_processor.py**  
  CSV 파일을 읽어 Power BI용 구조(히스토그램, 지도 등)로 변환합니다.
- **embedding_generator.py**  
  처리된 데이터 문서를 OpenAI Embeddings로 벡터화하여 FAISS 인덱스를 구축합니다.
- **inspect_similar_docs.py**  
  특정 문서와 유사도가 높은 문서를 확인하고 검토합니다.
- **rag_system_v*.py**  
  LangChain 기반 RAG 시스템의 버전별 예시 구현체입니다.
- **test_system.py**  
  전체 파이프라인(전처리→임베딩→QA)이 정상 동작하는지 자동으로 테스트합니다.
- **insight_generator.py**  
  분석 결과를 바탕으로 주요 인사이트를 생성하여 텍스트 파일로 출력합니다.

---

## 라이선스
이 프로젝트는 **MIT License** 하에 배포됩니다.  
자세한 내용은 [LICENSE](./LICENSE) 파일을 참조하세요.
