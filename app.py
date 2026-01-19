"""
RAGAS 평가 시스템 웹 UI
Streamlit을 사용한 RAG 시스템 평가 자동화 도구
"""

import json
import os
import shutil
import sys
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from datasets import Dataset

# 로컬 모듈 import
from askToLangflow import ask_to_langflow, load_payload, save_payload
from ragasEvaluator import (
    load_ragas_data,
    prepare_evaluation_data,
    run_ragas_evaluation,
    save_evaluation_results,
)
from responseParser import parse_response, save_parsed_response

# 페이지 설정
st.set_page_config(
    page_title="RAGAS 평가 시스템",
    page_icon="📊",
    layout="wide"
)

# 세션 상태 초기화
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'zip_files_to_cleanup' not in st.session_state:
    st.session_state.zip_files_to_cleanup = []

# 이전 세션의 ZIP 파일 정리 (옵션)
# 주의: 이전 세션의 ZIP 파일을 즉시 삭제하면 다운로드 불가능하므로
# 실제 운영 환경에서는 더 정교한 정리 전략 필요 (예: 타임아웃 기반)


def collect_query_data(
    query_data: Dict,
    langflow_url: str,
    langflow_api_key: str,
    temp_dir: Path
) -> Dict:
    """
    단일 쿼리에 대해 요청을 보내고 응답을 파싱하여 수집합니다.
    (평가는 하지 않음)
    
    Args:
        query_data: 쿼리 데이터 (query, ground_truth, gt_context)
        langflow_url: Langflow API URL
        langflow_api_key: Langflow API Key
        temp_dir: 임시 디렉토리
    
    Returns:
        수집된 데이터 딕셔너리 (parsed_data 포함)
    """
    try:
        # 1. 환경변수 설정
        os.environ['LANGFLOW_URL'] = langflow_url
        os.environ['LANGFLOW_API_KEY'] = langflow_api_key
        
        # 2. Payload 생성 (저장하지 않음)
        payload = {
            "input_value": query_data["query"],
            "output_type": "chat",
            "input_type": "chat"
        }
        
        # 3. Langflow API 요청
        # ask_to_langflow 호출 (자동으로 data/responses에 저장되고 저장 경로 반환)
        response_data, saved_response_filepath = ask_to_langflow(payload, url=langflow_url, return_filepath=True)
        
        # 저장된 응답 파일 읽기 (임시 디렉토리로 복사하지 않고 직접 읽기)
        saved_response_path = Path(saved_response_filepath)
        if saved_response_path.exists():
            # 저장된 전체 구조 읽기 (parse_response에 전달하기 위해)
            with open(saved_response_path, 'r', encoding='utf-8') as f:
                saved_response_data = json.load(f)
        else:
            # 파일이 없는 경우 response_data를 사용 (fallback)
            saved_response_data = {
                'response': response_data
            }
        
        # 4. 응답 파싱
        # saved_response_data는 {'timestamp': ..., 'url': ..., 'payload': ..., 'response': ...} 구조
        parsed_data = parse_response(saved_response_data)
        
        # 5. 파싱된 응답 저장
        parsed_path = temp_dir / "parsed_responses" / f"parsed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.json"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        save_parsed_response(parsed_data, str(parsed_path))
        
        return {
            'query': query_data["query"],
            'success': True,
            'query_data': query_data,
            'parsed_data': parsed_data,
            'parsed_path': str(parsed_path),
            'response_path': str(saved_response_path)  # 영구 저장된 응답 파일 경로 추가
        }
        
    except Exception as e:
        return {
            'query': query_data["query"],
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }


def prepare_batch_evaluation_data(
    collected_data: List[Dict]
) -> Tuple[Dataset, List[Dict]]:
    """
    수집된 모든 데이터를 RAGAS 평가용 Dataset으로 준비합니다.
    답변이 없거나 context가 null인 케이스는 제외합니다.
    
    Args:
        collected_data: 수집된 데이터 리스트 (각 항목은 query_data와 parsed_data 포함)
    
    Returns:
        tuple: (Dataset, 평가에 포함된 데이터 리스트)
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    ground_truths_list = []
    included_items = []  # 평가에 포함된 항목들
    
    for item in collected_data:
        if not item['success']:
            continue
        
        parsed_data = item.get('parsed_data', {})
        answer = parsed_data.get('answer', '')
        context = parsed_data.get('context', '')
        
        # 평가에서 제외할 케이스 체크
        # 1. 답변이 없는 경우
        if not answer or answer.strip() == '':
            continue
        
        # 2. 답변은 있지만 context가 null이거나 빈 문자열인 경우
        if not context or context.strip() == '':
            continue
        
        # 평가에 포함
        questions.append(item['query_data']['query'])
        answers.append(answer)
        contexts.append([context])
        ground_truths.append(item['query_data']['ground_truth'])
        ground_truths_list.append([item['query_data']['gt_context']])
        included_items.append(item)
    
    if len(questions) == 0:
        raise ValueError("평가할 수 있는 유효한 데이터가 없습니다. (답변이 없거나 context가 null인 케이스는 제외됩니다)")
    
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "ground_truths": ground_truths_list,
    }
    
    from datasets import Dataset
    dataset = Dataset.from_dict(data_dict)
    return dataset, included_items


def create_zip_file(results_dir: Path, output_path: Path, collected_data: List[Dict] = None):
    """
    결과 디렉토리의 모든 파일을 ZIP 파일로 압축합니다.
    영구 저장된 파일들도 포함합니다 (collected_data가 제공된 경우).
    
    Args:
        results_dir: 압축할 디렉토리
        output_path: 출력 ZIP 파일 경로
        collected_data: 수집된 데이터 리스트 (영구 저장된 파일 경로 포함)
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 결과 디렉토리의 파일들 추가
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = Path(root) / file
                # results_dir 기준으로 상대 경로 생성
                arcname = file_path.relative_to(results_dir)
                zipf.write(file_path, arcname)
        
        # 영구 저장된 파일들도 ZIP에 추가 (이중 저장 최적화: 복사 없이 ZIP에 직접 추가)
        if collected_data:
            for item in collected_data:
                if item.get('success'):
                    # 영구 저장된 응답 파일 추가
                    if 'response_path' in item:
                        response_path = Path(item['response_path'])
                        if response_path.exists():
                            zipf.write(response_path, f"data/responses/{response_path.name}")
                    
                    # 파싱된 응답 파일 추가 (임시 디렉토리에 있는 것)
                    if 'parsed_path' in item:
                        parsed_path = Path(item['parsed_path'])
                        if parsed_path.exists():
                            zipf.write(parsed_path, f"data/parsed_responses/{parsed_path.name}")


def main():
    st.title("📊 RAGAS 평가 시스템")
    st.markdown("---")
    
    # 사이드바: 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "쿼리셋 JSON 파일 업로드",
            type=['json'],
            help="data.json 형식의 쿼리셋 파일을 업로드하세요"
        )
        
        # URL 및 API 키 입력
        langflow_url = st.text_input(
            "Langflow URL",
            value=os.getenv('LANGFLOW_URL', ''),
            help="Langflow API URL (예: http://10.1.1.70:7860/api/v1/run/...)",
            type="default"
        )
        
        langflow_api_key = st.text_input(
            "Langflow API Key",
            value=os.getenv('LANGFLOW_API_KEY', ''),
            type="password",
            help="Langflow API Key"
        )
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv('OPENAI_API_KEY', ''),
            type="password",
            help="OpenAI API Key (RAGAS 평가에 사용)"
        )
        
        st.markdown("---")
        
        # 평가 시작 버튼
        start_button = st.button(
            "🚀 평가 시작",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.is_running
        )
        
        # 평가 중지 버튼
        if st.session_state.is_running:
            stop_button = st.button(
                "⏹️ 평가 중지",
                type="secondary",
                use_container_width=True
            )
            if stop_button:
                st.session_state.is_running = False
                st.rerun()
    
    # 메인 영역
    if uploaded_file is None:
        st.info("👈 사이드바에서 쿼리셋 JSON 파일을 업로드하고 설정을 입력한 후 평가를 시작하세요.")
        
        # 예시 데이터 표시
        with st.expander("📝 쿼리셋 파일 형식 예시"):
            example_data = [
                {
                    "query": "계약규정의 제정 목적은 무엇인가?",
                    "ground_truth": "이 규정은 회사의 계약업무 처리에 필요한 기본사항을 정하여 계약업무의 원활한 수행을 도모하는 것을 목적으로 한다.",
                    "gt_context": "계약규정 제1조(목적)"
                }
            ]
            st.json(example_data)
    else:
        # 업로드된 파일 읽기
        try:
            file_content = uploaded_file.read()
            queries_data = json.loads(file_content.decode('utf-8'))
            
            if not isinstance(queries_data, list):
                st.error("❌ JSON 파일은 배열 형식이어야 합니다.")
                st.stop()
            
            st.success(f"✅ {len(queries_data)}개의 쿼리를 불러왔습니다.")
            
            # 쿼리 목록 표시
            with st.expander(f"📋 쿼리 목록 ({len(queries_data)}개)"):
                for i, query_data in enumerate(queries_data, 1):
                    st.markdown(f"**{i}. {query_data.get('query', 'N/A')}**")
            
            # 평가 시작
            if start_button:
                # 입력 검증
                if not langflow_url:
                    st.error("❌ Langflow URL을 입력하세요.")
                    st.stop()
                if not langflow_api_key:
                    st.error("❌ Langflow API Key를 입력하세요.")
                    st.stop()
                if not openai_api_key:
                    st.error("❌ OpenAI API Key를 입력하세요.")
                    st.stop()
                
                # 평가 시작
                st.session_state.is_running = True
                st.session_state.evaluation_results = []
                st.session_state.progress = 0
                st.session_state.total_queries = len(queries_data)
                
                # 임시 디렉토리 생성
                temp_dir = Path(tempfile.mkdtemp(prefix="ragas_eval_"))
                zip_path = None  # ZIP 파일 경로 추적용
                
                try:
                    # 진행도 표시 영역
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    
                    # 환경변수 설정
                    os.environ['LANGFLOW_URL'] = langflow_url
                    os.environ['LANGFLOW_API_KEY'] = langflow_api_key
                    os.environ['OPENAI_API_KEY'] = openai_api_key
                    
                    # 1단계: 모든 쿼리에 대해 데이터 수집
                    status_text.text("📥 데이터 수집 중...")
                    collected_data = []
                
                for idx, query_data in enumerate(queries_data):
                    if not st.session_state.is_running:
                        st.warning("⚠️ 평가가 중지되었습니다.")
                        break
                    
                    # 진행도 업데이트 (데이터 수집 단계: 0~70%)
                    progress = (idx + 1) / len(queries_data) * 0.7
                    progress_bar.progress(progress)
                    status_text.text(f"데이터 수집 중: {idx + 1}/{len(queries_data)} - {query_data.get('query', 'N/A')[:50]}...")
                    
                    # 데이터 수집
                    result = collect_query_data(
                        query_data,
                        langflow_url,
                        langflow_api_key,
                        temp_dir
                    )
                    
                    collected_data.append(result)
                    
                    # 결과 표시
                    with results_container:
                        if result['success']:
                            st.success(f"✅ {result['query'][:50]}... (수집 완료)")
                        else:
                            st.error(f"❌ {result['query'][:50]}... - {result.get('error', 'Unknown error')}")
                
                # 2단계: 모든 데이터 수집 완료 후 평가 실행
                if st.session_state.is_running:
                    status_text.text("📊 평가 실행 중...")
                    progress_bar.progress(0.75)
                    
                    # 성공한 데이터만 필터링
                    successful_data = [item for item in collected_data if item['success']]
                    
                    if len(successful_data) == 0:
                        st.error("❌ 평가할 수 있는 데이터가 없습니다.")
                        st.session_state.is_running = False
                    else:
                        try:
                            # 배치 평가 데이터 준비 (답변이 없거나 context가 null인 케이스 제외)
                            progress_bar.progress(0.8)
                            dataset, included_items = prepare_batch_evaluation_data(successful_data)
                            
                            # 제외된 케이스 확인
                            excluded_items = [item for item in successful_data if item not in included_items]
                            if excluded_items:
                                excluded_queries = [item['query'] for item in excluded_items]
                                st.warning(f"⚠️ 다음 {len(excluded_items)}개 쿼리는 답변이 없거나 context가 null이어서 평가에서 제외되었습니다:")
                                for query in excluded_queries:
                                    st.caption(f"  - {query[:60]}...")
                            
                            # RAGAS 평가 실행 (전체 DataFrame 얻기)
                            from ragas import evaluate
                            from ragas.metrics import (
                                faithfulness,
                                answer_relevancy,
                                context_precision,
                                context_recall,
                            )
                            from ragas.llms import llm_factory
                            from openai import OpenAI
                            
                            # OpenAI 클라이언트 및 LLM 설정
                            api_key = os.getenv('OPENAI_API_KEY')
                            openai_client = OpenAI(api_key=api_key)
                            llm = llm_factory("gpt-4o-mini", client=openai_client)
                            
                            # 평가 메트릭 설정
                            metrics = [
                                faithfulness.__class__(llm=llm),
                                answer_relevancy.__class__(llm=llm),
                                context_precision.__class__(llm=llm),
                                context_recall.__class__(llm=llm),
                            ]
                            
                            # 전체 평가 실행
                            progress_bar.progress(0.9)
                            full_result = evaluate(dataset=dataset, metrics=metrics)
                            results_df = full_result.to_pandas()
                            
                            # 각 쿼리별로 결과 저장
                            results_dir = temp_dir / "results"
                            results_dir.mkdir(parents=True, exist_ok=True)
                            
                            evaluation_results_list = []
                            
                            # 평가에 포함된 항목들에 대해 결과 매핑
                            for idx, item in enumerate(included_items):
                                if idx < len(results_df):
                                    row = results_df.iloc[idx]
                                    result_dict = {
                                        'faithfulness': float(row.get('faithfulness', 0)) if pd.notna(row.get('faithfulness', 0)) else 0,
                                        'answer_relevancy': float(row.get('answer_relevancy', 0)) if pd.notna(row.get('answer_relevancy', 0)) else 0,
                                        'context_precision': float(row.get('context_precision', 0)) if pd.notna(row.get('context_precision', 0)) else 0,
                                        'context_recall': float(row.get('context_recall', 0)) if pd.notna(row.get('context_recall', 0)) else 0,
                                    }
                                else:
                                    # 기본값 사용
                                    result_dict = {
                                        'faithfulness': 0,
                                        'answer_relevancy': 0,
                                        'context_precision': 0,
                                        'context_recall': 0,
                                    }
                                
                                # 개별 결과 저장
                                json_path = save_evaluation_results(
                                    result_dict,
                                    [item['query_data']],
                                    item['parsed_data'],
                                    str(results_dir)
                                )
                                
                                evaluation_results_list.append({
                                    'query': item['query'],
                                    'success': True,
                                    'results': result_dict,
                                    'json_path': json_path,
                                    'parsed_path': item.get('parsed_path')
                                })
                            
                            # 평가에서 제외된 항목들 (답변이 없거나 context가 null)
                            for item in excluded_items:
                                evaluation_results_list.append({
                                    'query': item['query'],
                                    'success': False,
                                    'error': '답변이 없거나 context가 null이어서 평가에서 제외됨',
                                    'error_type': 'ExcludedFromEvaluation'
                                })
                            
                            # 실패한 쿼리도 결과에 포함
                            for item in collected_data:
                                if not item['success']:
                                    evaluation_results_list.append(item)
                            
                            st.session_state.evaluation_results = evaluation_results_list
                            
                            # 평균 점수 계산 및 저장
                            successful_eval_results = [r for r in evaluation_results_list if r['success']]
                            if successful_eval_results:
                                metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
                                avg_scores = {}
                                for metric in metrics:
                                    scores = []
                                    for r in successful_eval_results:
                                        if 'results' in r and metric in r['results']:
                                            score = r['results'][metric]
                                            if isinstance(score, (int, float)) and not (isinstance(score, float) and score != score):  # NaN 체크
                                                scores.append(score)
                                    if scores:
                                        avg_scores[metric] = sum(scores) / len(scores)
                                
                                # 평균 점수를 JSON 파일로 저장
                                if avg_scores:
                                    avg_results_data = {
                                        "timestamp": datetime.now().isoformat(),
                                        "summary": {
                                            "total_queries": len(evaluation_results_list),
                                            "successful_queries": len(successful_eval_results),
                                            "failed_queries": len(evaluation_results_list) - len(successful_eval_results)
                                        },
                                        "average_scores": avg_scores
                                    }
                                    
                                    avg_json_filepath = results_dir / "average_scores.json"
                                    with open(avg_json_filepath, 'w', encoding='utf-8') as f:
                                        json.dump(avg_results_data, f, ensure_ascii=False, indent=2)
                                    
                                    print(f"평균 점수가 저장되었습니다: {avg_json_filepath}")
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ 모든 평가가 완료되었습니다!")
                            
                        except ValueError as e:
                            # 평가할 수 있는 유효한 데이터가 없는 경우
                            st.error(f"❌ {str(e)}")
                            st.session_state.evaluation_results = collected_data
                            st.session_state.is_running = False
                        except Exception as e:
                            st.error(f"❌ 평가 실행 중 오류 발생: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.session_state.evaluation_results = collected_data
                
                # 평가 완료
                if st.session_state.is_running:
                    st.session_state.is_running = False
                    
                    # 결과 요약
                    st.markdown("---")
                    st.header("📊 평가 결과 요약")
                    
                    successful = sum(1 for r in st.session_state.evaluation_results if r['success'])
                    failed = len(st.session_state.evaluation_results) - successful
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("전체 쿼리", len(st.session_state.evaluation_results))
                    with col2:
                        st.metric("성공", successful, delta=f"{successful/len(st.session_state.evaluation_results)*100:.1f}%")
                    with col3:
                        st.metric("실패", failed, delta=f"-{failed/len(st.session_state.evaluation_results)*100:.1f}%")
                    
                    # 성공한 평가의 평균 점수 계산
                    if successful > 0:
                        successful_results = [r for r in st.session_state.evaluation_results if r['success']]
                        metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
                        
                        st.markdown("### 📈 평균 평가 점수")
                        avg_scores = {}
                        for metric in metrics:
                            scores = []
                            for r in successful_results:
                                if 'results' in r and metric in r['results']:
                                    score = r['results'][metric]
                                    if isinstance(score, (int, float)) and not (isinstance(score, float) and score != score):  # NaN 체크
                                        scores.append(score)
                            if scores:
                                avg_scores[metric] = sum(scores) / len(scores)
                        
                        if avg_scores:
                            cols = st.columns(len(avg_scores))
                            for i, (metric, score) in enumerate(avg_scores.items()):
                                with cols[i]:
                                    st.metric(metric.replace('_', ' ').title(), f"{score:.4f}")
                    
                    # 압축 파일 생성 및 다운로드
                    st.markdown("---")
                    st.header("📦 결과 다운로드")
                    
                    zip_filename = f"ragas_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.zip"
                    zip_path = temp_dir.parent / zip_filename
                    
                    try:
                        create_zip_file(temp_dir, zip_path, collected_data)
                        
                        # ZIP 파일 읽기
                        zip_data = None
                        with open(zip_path, 'rb') as f:
                            zip_data = f.read()
                        
                        if zip_data:
                            st.download_button(
                                label="📥 결과 파일 다운로드 (ZIP)",
                                data=zip_data,
                                file_name=zip_filename,
                                mime="application/zip",
                                use_container_width=True
                            )
                            
                            st.success(f"✅ 압축 파일이 생성되었습니다: {zip_filename}")
                            
                            # 파일 크기 표시
                            file_size = len(zip_data) / (1024 * 1024)  # MB
                            st.caption(f"파일 크기: {file_size:.2f} MB")
                        else:
                            st.error("❌ 압축 파일 생성 실패")
                        
                    except Exception as e:
                        st.error(f"❌ 압축 파일 생성 중 오류: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                    
                    # 상세 결과 표시
                    with st.expander("📋 상세 결과"):
                        for i, result in enumerate(st.session_state.evaluation_results, 1):
                            if result['success']:
                                st.markdown(f"#### {i}. {result['query']}")
                                st.json(result['results'])
                            else:
                                st.markdown(f"#### {i}. {result['query']} ❌")
                                st.error(f"오류: {result.get('error', 'Unknown error')}")
                
                finally:
                    # 임시 디렉토리 및 ZIP 파일 정리
                    try:
                        if temp_dir.exists():
                            shutil.rmtree(temp_dir)
                            print(f"임시 디렉토리 정리 완료: {temp_dir}")
                    except Exception as e:
                        print(f"임시 디렉토리 정리 중 오류 (무시 가능): {e}")
                    
                    try:
                        if zip_path and zip_path.exists():
                            # ZIP 파일은 다운로드 후 일정 시간 후 삭제하는 것이 좋지만,
                            # Streamlit에서는 세션별로 관리하기 어려우므로 
                            # 최소한 세션 상태에 경로를 저장하고 다음 페이지 로드 시 정리할 수 있도록 함
                            # 여기서는 즉시 삭제하지 않고 남겨둠 (사용자가 다시 다운로드할 수 있도록)
                            # 대신 ZIP 파일 경로를 세션 상태에 저장
                            if 'zip_files_to_cleanup' not in st.session_state:
                                st.session_state.zip_files_to_cleanup = []
                            st.session_state.zip_files_to_cleanup.append(str(zip_path))
                    except Exception as e:
                        print(f"ZIP 파일 정리 중 오류 (무시 가능): {e}")
                
        except json.JSONDecodeError:
            st.error("❌ JSON 파일 형식이 올바르지 않습니다.")
        except Exception as e:
            st.error(f"❌ 오류 발생: {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == '__main__':
    main()

