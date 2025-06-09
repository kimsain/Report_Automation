"""
LangChain 기반 인사이트 생성 엔진

이 모듈은 LangChain의 ChatOpenAI와 프롬프트 템플릿을 활용하여 
데이터 기반 인사이트를 자동으로 생성하는 기능을 제공합니다.
"""

import os
import re
import time
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain

from langchain_config import LangChainConfig
from langchain_embedding_utils import LangChainEmbeddingUtils

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = LangChainConfig.OPENAI_API_KEY

class LangChainInsightGenerator:
    """
    LangChain 기반 인사이트 생성 클래스
    """
    
    def __init__(self):
        """인사이트 생성기 초기화"""
        # LangChain ChatOpenAI 모델 초기화
        self.search_llm = ChatOpenAI(
            model_name=LangChainConfig.OPENAI_SEARCH_MODEL,
            temperature=0.5,
            openai_api_key=LangChainConfig.OPENAI_API_KEY
        )
        
        self.evaluation_llm = ChatOpenAI(
            model_name=LangChainConfig.OPENAI_COMPLETION_MODEL,
            temperature=0.3,
            openai_api_key=LangChainConfig.OPENAI_API_KEY
        )
        
        # 임베딩 유틸리티
        self.embedding_utils = LangChainEmbeddingUtils()
        
        # 프롬프트 템플릿 설정
        self._setup_prompt_templates()
    
    def _setup_prompt_templates(self):
        """프롬프트 템플릿 설정"""
        # 인사이트 생성용 시스템 프롬프트
        self.insight_system_template = """
        당신은 데이터 분석 전문가입니다.
        주어진 지역 현황 정보를 바탕으로 해당 현상이 발생한 원인을 분석하고 설명하세요.

        # 웹 검색을 통해 확인 가능한 {year}년 {month}월을 포함하여, 직전 3개월(과거 3개월) 정보만 활용하세요.
        # 응답은 다음 형식을 따르세요:
            - 불렛포인트 {insight_count}개로 작성할 것.
            - 각 포인트는 2~3문장으로 구성할 것.
            - 각 원인 설명에는 참고한 출처(URL 포함)를 명시할 것.
            - 불렛포인트외 다른 문장은 응답에 포함하지 말 것.
        # 응답은 중립적이고 객관적인 톤을 유지할 것.
        """
        
        # 인사이트 평가용 시스템 프롬프트
        self.evaluation_system_template = """
        당신은 데이터 분석 전문가이며, 인사이트 평가 전문가입니다.
        {crym}년 {sido} {sigungu}의 {subject}에 대한 인사이트를 평가합니다.
        우선, 주어진 데이터 정보는 아래와 같으며, 인사이트를 평가해주세요

        {query_text}
        """
        
        # 프롬프트 템플릿 생성
        self.insight_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.insight_system_template),
            HumanMessagePromptTemplate.from_template("{query_text}")
        ])
        
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.evaluation_system_template),
            HumanMessagePromptTemplate.from_template("""
            다음 인사이트의 정보 밀도와 실용성을 평가하고, 1~10점으로 점수를 매기세요.
            인사이트: "{insight}"
            단, 점수만 숫자로 출력하세요.
            """)
        ])
    
    def generate_insights_from_data(self, 
                                   crym: str, 
                                   sido: str, 
                                   sigungu: str, 
                                   subject: str,
                                   data_statements: List[str]) -> pd.DataFrame:
        """
        데이터로부터 인사이트 생성
        
        Args:
            crym (str): 연월 (예: "202502")
            sido (str): 시도명
            sigungu (str): 시군구명
            subject (str): 주제
            data_statements (List[str]): 데이터 설명 문장 리스트
        
        Returns:
            pd.DataFrame: 생성된 인사이트와 점수
        """
        # 연월 파싱
        year = crym[:4]
        month = crym[4:6].lstrip("0")
        
        # 쿼리 텍스트 생성
        query_text = f"{year}년 {month}월 {sido} {sigungu}의 현황 정보:\n"
        for stmt in data_statements:
            query_text += f" - {stmt}\n"
        
        # 인사이트 생성
        insights_raw = self._generate_raw_insights(year, month, query_text)
        
        # 인사이트 파싱 및 점수 계산
        return self._process_insights(insights_raw, crym, sido, sigungu, subject, query_text)
    
    def _generate_raw_insights(self, year: str, month: str, query_text: str) -> str:
        """원시 인사이트 생성"""
        try:
            # 프롬프트 포맷팅
            messages = self.insight_prompt.format_messages(
                year=year,
                month=month,
                insight_count=LangChainConfig.INSIGHT_COUNT,
                query_text=query_text
            )
            
            # LLM 호출 (웹 검색 기능 포함)
            response = self.search_llm(messages)
            return response.content
            
        except Exception as e:
            print(f"인사이트 생성 중 오류 발생: {e}")
            return ""\n    
    def _process_insights(self, 
                         content_raw: str, 
                         crym: str, 
                         sido: str, 
                         sigungu: str, 
                         subject: str,
                         query_text: str) -> pd.DataFrame:
        """인사이트 처리 및 점수 계산"""
        # 인사이트 파싱
        insights_raw = content_raw.strip().split('\n\n')
        insight_texts, insight_urls = [], []
        
        for item in insights_raw:
            # URL 추출
            url_match = re.search(r'\((https?://.*?)\)', item)
            url = url_match.group(1) if url_match else None
            
            # 텍스트 정리
            text_cleaned = re.sub(r'\[.*?\]\(.*?\)', '', item)
            text_cleaned = text_cleaned.replace("- ", "").strip()
            
            insight_texts.append(text_cleaned)
            insight_urls.append(url)
        
        # TF-IDF 점수 계산
        tfidf_scores = self._calculate_tfidf_scores(insight_texts)
        
        # GPT 평가 점수 계산
        gpt_scores = self._calculate_gpt_scores(insight_texts, crym, sido, sigungu, subject, query_text)
        
        # 최종 점수 계산
        final_scores = (LangChainConfig.TFIDF_WEIGHT * tfidf_scores + 
                       LangChainConfig.GPT_WEIGHT * gpt_scores)
        
        # DataFrame 생성
        df_result = pd.DataFrame({
            'crym': [crym] * len(insight_texts),
            'sido': [sido] * len(insight_texts),
            'sigungu': [sigungu] * len(insight_texts),
            'subject': [subject] * len(insight_texts),
            'insight_no': list(range(1, len(insight_texts) + 1)),
            'insight': insight_texts,
            'tfidf_score': tfidf_scores,
            'gpt_score': gpt_scores,
            'final_score': final_scores,
            'url': insight_urls
        })
        
        return df_result.sort_values(by='final_score', ascending=False)
    
    def _calculate_tfidf_scores(self, texts: List[str]) -> np.ndarray:
        """TF-IDF 점수 계산"""
        try:
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts)
            return X.sum(axis=1).A1
        except Exception as e:
            print(f"TF-IDF 점수 계산 중 오류 발생: {e}")
            return np.ones(len(texts)) * 5.0
    
    def _calculate_gpt_scores(self, 
                             texts: List[str], 
                             crym: str, 
                             sido: str, 
                             sigungu: str, 
                             subject: str,
                             query_text: str) -> np.ndarray:
        """GPT 평가 점수 계산"""
        scores = []
        
        for text in texts:
            try:
                # 프롬프트 포맷팅
                messages = self.evaluation_prompt.format_messages(
                    crym=crym,
                    sido=sido,
                    sigungu=sigungu,
                    subject=subject,
                    query_text=query_text,
                    insight=text
                )
                
                # LLM 호출
                response = self.evaluation_llm(messages)
                score_text = response.content.strip()
                
                # 점수 파싱
                try:
                    score = float(score_text)
                except ValueError:
                    score = 5.0
                
                scores.append(score)
                time.sleep(2)  # Rate limit 대응
                
            except Exception as e:
                print(f"GPT 점수 계산 중 오류 발생: {e}")
                scores.append(5.0)
        
        return np.array(scores)
    
    def generate_insights_from_csv(self, csv_file_path: str, output_file_path: str = None):
        """CSV 파일로부터 인사이트 생성"""
        try:
            # CSV 파일 읽기
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            
            # 결과 저장용 DataFrame
            df_insight = pd.DataFrame()
            
            # 고유한 그룹 조합 추출
            group_keys = df[['crym', 'sido', 'sigungu', 'subject']].drop_duplicates()
            
            for _, group in group_keys.iterrows():
                crym, sido, sigungu, subject = group['crym'], group['sido'], group['sigungu'], group['subject']
                
                # 해당 그룹의 데이터 필터링
                filtered_data = df[
                    (df['crym'] == crym) &
                    (df['sido'] == sido) &
                    (df['sigungu'] == sigungu) &
                    (df['subject'] == subject)
                ]
                
                # 데이터 설명 문장 추출
                data_statements = filtered_data['stmt'].tolist()
                
                # 인사이트 생성
                temp_df = self.generate_insights_from_data(crym, sido, sigungu, subject, data_statements)
                df_insight = pd.concat([df_insight, temp_df], ignore_index=True)
            
            # 결과 저장
            if output_file_path:
                df_insight.to_csv(output_file_path, index=False, encoding='utf-8')
                print(f"인사이트 결과가 {output_file_path}에 저장되었습니다.")
            
            return df_insight
            
        except Exception as e:
            print(f"CSV 인사이트 생성 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def search_related_insights(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """관련 인사이트 검색"""
        return self.embedding_utils.search_similar_documents(query, k)

if __name__ == "__main__":
    # 테스트 코드
    generator = LangChainInsightGenerator()
    
    # 테스트 데이터
    test_statements = [
        "수원시의 당월 전체 신규 가맹점 수는 전월 대비 감소",
        "휴·폐업 가맹점은 증가하여 전반적으로 사업자 유입 흐름이 감소",
        "일반의원 업종에서 09-11시간대에 최대 매출액이 발생",
        "편의점 업종에서 19-21시간대에 최다 매출건수가 발생"
    ]
    
    result = generator.generate_insights_from_data(
        crym="202502",
        sido="경기도",
        sigungu="수원시",
        subject="상권분석",
        data_statements=test_statements
    )
    
    print("생성된 인사이트:")
    print(result.head())

