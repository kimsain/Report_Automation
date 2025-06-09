#!/usr/bin/env python3
"""
LangChain 기반 보고서 자동화 시스템 메인 실행 스크립트

이 스크립트는 LangChain MVP 시스템의 주요 기능들을 실행할 수 있는 통합 인터페이스를 제공합니다.
"""

import sys
import argparse
from langchain_database_manager import LangChainDatabaseManager
from langchain_cli import main as cli_main

def setup_database():
    """데이터베이스 초기 설정"""
    print("🔧 LangChain 기반 보고서 자동화 시스템 데이터베이스 설정을 시작합니다...")
    manager = LangChainDatabaseManager()
    manager.setup_complete_database()
    print("✅ 데이터베이스 설정이 완료되었습니다!")

def run_cli():
    """CLI 실행"""
    print("💻 LangChain 기반 보고서 자동화 시스템 CLI를 시작합니다...")
    # sys.argv를 조작하여 CLI 인자를 전달
    original_argv = sys.argv.copy()
    sys.argv = sys.argv[2:]  # 'main.py run-cli' 부분 제거
    try:
        cli_main()
    finally:
        sys.argv = original_argv

def run_web_app():
    """웹 애플리케이션 실행"""
    print("🌐 LangChain 기반 보고서 자동화 시스템 웹 애플리케이션을 시작합니다...")
    from langchain_app import app, initialize_components
    
    # 컴포넌트 초기화
    initialize_components()
    
    # Flask 앱 실행
    app.run(host='0.0.0.0', port=5000, debug=True)

def show_status():
    """시스템 상태 확인"""
    print("📊 LangChain 기반 보고서 자동화 시스템 상태:")
    print("- 설정 파일: langchain_config.py")
    print("- 데이터베이스 매니저: langchain_database_manager.py")
    print("- 임베딩 유틸리티: langchain_embedding_utils.py")
    print("- 벡터 DB 매니저: langchain_vector_db_manager.py")
    print("- 텍스트 스플리터: langchain_text_splitter.py")
    print("- RAG 시스템: langchain_rag_system.py")
    print("- 인사이트 생성기: langchain_insight_generator.py")
    print("- Flask 애플리케이션: langchain_app.py")
    print("- CLI: langchain_cli.py")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LangChain 기반 보고서 자동화 시스템")
    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령어")
    
    # 데이터베이스 설정 명령어
    setup_parser = subparsers.add_parser("setup", help="데이터베이스 초기 설정")
    
    # CLI 실행 명령어
    cli_parser = subparsers.add_parser("cli", help="CLI 실행")
    
    # 웹 앱 실행 명령어
    web_parser = subparsers.add_parser("web", help="웹 애플리케이션 실행")
    
    # 상태 확인 명령어
    status_parser = subparsers.add_parser("status", help="시스템 상태 확인")
    
    args, unknown = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "setup":
            setup_database()
        elif args.command == "cli":
            # 남은 인자들을 CLI로 전달
            sys.argv = ["langchain_cli.py"] + unknown
            run_cli()
        elif args.command == "web":
            run_web_app()
        elif args.command == "status":
            show_status()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

