"""
RAGAS 평가 시스템 웹 UI
Streamlit을 사용한 RAG 시스템 평가 자동화 도구
"""

import json
import os
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from datasets import Dataset

# Streamlit의 ThreadPoolExecutor 경고 무시 설정
import warnings
import logging

# Streamlit의 ScriptRunContext 경고 무시
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
logging.getLogger('streamlit.runtime.scriptrunner.script_runner').setLevel(logging.ERROR)
# Streamlit의 모든 경고 로거 레벨 조정
for logger_name in ['streamlit', 'streamlit.runtime', 'streamlit.runtime.scriptrunner']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

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
if 'langflow_url' not in st.session_state:
    st.session_state.langflow_url = 'http://10.1.1.70:7860/api/v1/run/da37679a-32c8-4fd4-a650-3b98e155ccf8?stream=true'

# 이전 세션의 ZIP 파일 정리 (옵션)
# 주의: 이전 세션의 ZIP 파일을 즉시 삭제하면 다운로드 불가능하므로
# 실제 운영 환경에서는 더 정교한 정리 전략 필요 (예: 타임아웃 기반)


def collect_query_data(
    query_data: Dict,
    langflow_url: str,
    langflow_api_key: str,
    temp_dir: Path,
    session_id: str = None,
    timer_callback: callable = None,
    query_index: int = None
) -> Dict:
    """
    단일 쿼리에 대해 요청을 보내고 응답을 파싱하여 수집합니다.
    (평가는 하지 않음)
    
    Args:
        query_data: 쿼리 데이터 (query, ground_truth, gt_context)
        langflow_url: Langflow API URL
        langflow_api_key: Langflow API Key
        temp_dir: 임시 디렉토리
        session_id: Langflow 채팅 세션 ID (선택사항)
        timer_callback: 타이머 업데이트 콜백 함수 (선택사항)
    
    Returns:
        수집된 데이터 딕셔너리 (parsed_data 포함)
    """
    # NoSessionContext 예외를 무시하고 실제 작업 수행
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
        
        # session_id가 제공된 경우 payload에 추가
        if session_id:
            payload["session_id"] = session_id
        
        # 타이머 콜백은 스레드 내에서 호출하지 않음 (Streamlit 컨텍스트 문제 방지)
        # timer_callback은 메인 스레드에서만 호출됨
        
        # 3. Langflow API 요청
        # ask_to_langflow 호출 (자동으로 data/responses에 저장되고 저장 경로 반환)
        # NoSessionContext 예외를 무시하고 계속 진행
        try:
            response_data, saved_response_filepath = ask_to_langflow(payload, url=langflow_url, return_filepath=True)
        except Exception as api_e:
            # NoSessionContext는 무시하고 실제 오류만 처리
            error_type = type(api_e).__name__
            if error_type != 'NoSessionContext':
                raise  # NoSessionContext가 아닌 실제 오류는 다시 발생시킴
            # NoSessionContext인 경우 재시도
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
        try:
            parsed_data = parse_response(saved_response_data)
        except Exception as parse_e:
            # NoSessionContext는 무시
            if type(parse_e).__name__ != 'NoSessionContext':
                raise
            parsed_data = parse_response(saved_response_data)
        
        # 5. 파싱된 응답 저장
        parsed_path = temp_dir / "parsed_responses" / f"parsed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.json"
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_parsed_response(parsed_data, str(parsed_path))
        except Exception as save_e:
            # NoSessionContext는 무시
            if type(save_e).__name__ != 'NoSessionContext':
                raise
            save_parsed_response(parsed_data, str(parsed_path))
        
        return {
            'query': query_data["query"],
            'success': True,
            'query_data': query_data,
            'parsed_data': parsed_data,
            'parsed_path': str(parsed_path),
            'response_path': str(saved_response_path),  # 영구 저장된 응답 파일 경로 추가
            'session_id': session_id if session_id else None  # 세션 ID 추가
        }
        
    except Exception as e:
        # NoSessionContext는 실제 오류가 아니므로 무시
        error_type = type(e).__name__
        if error_type == 'NoSessionContext':
            # NoSessionContext인 경우 빈 오류 메시지 대신 명확한 메시지 제공
            return {
                'query': query_data["query"],
                'success': False,
                'error': 'Streamlit 컨텍스트 오류 (NoSessionContext) - 스레드 내 실행 중 발생한 것으로 정상 동작에 영향 없음',
                'error_type': error_type
            }
        return {
            'query': query_data["query"],
            'success': False,
            'error': str(e) if str(e) else repr(e),
            'error_type': error_type
        }


def collect_query_data_batch(
    queries_data: List[Dict],
    langflow_url: str,
    langflow_api_key: str,
    temp_dir: Path,
    eval_start_str: str,
    batch_size: int = 5,
    timer_callback: callable = None,
    progress_callback: callable = None,
    results_container = None  # 사용하지 않음 (스레드 내 Streamlit 호출 방지)
) -> List[Dict]:
    """
    여러 쿼리를 배치로 처리하여 데이터를 수집합니다.
    
    Args:
        queries_data: 쿼리 데이터 리스트
        langflow_url: Langflow API URL
        langflow_api_key: Langflow API Key
        temp_dir: 임시 디렉토리
        eval_start_str: 평가 시작 시간 문자열 (세션 ID 생성용)
        batch_size: 동시 처리할 배치 크기 (기본값: 5)
        timer_callback: 타이머 업데이트 콜백 함수
        progress_callback: 진행 상황 업데이트 콜백 함수
        results_container: 결과 표시용 Streamlit 컨테이너
    
    Returns:
        수집된 데이터 리스트
    """
    collected_data = []
    total_queries = len(queries_data)
    
    # 시간 포맷팅 함수 (스레드 내에서도 사용 가능하도록 외부에 정의)
    def format_time(seconds):
        """초 단위 시간을 읽기 쉬운 형식으로 변환"""
        if seconds < 0:
            return "0초"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        if hours > 0:
            return f"{hours}시간 {minutes}분 {secs}초"
        elif minutes > 0:
            return f"{minutes}분 {secs}초"
        elif secs > 0:
            return f"{secs}.{millisecs//100}초"
        else:
            return f"{millisecs}ms"
    
    def process_single_query(query_data_with_index: Tuple[int, Dict]) -> Tuple[int, Dict]:
        """단일 쿼리 처리 함수 (ThreadPoolExecutor에서 사용)"""
        idx, query_data = query_data_with_index
        query_number = idx + 1
        session_id = f"Q_{eval_start_str}_{query_number:03d}"
        
        query_start_time = time.time()
        
        # 병렬 처리 확인을 위한 로깅 (디버깅용)
        import threading
        thread_id = threading.current_thread().ident
        thread_name = threading.current_thread().name
        start_time_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[DEBUG] 쿼리 {idx+1} 시작 | 스레드: {thread_name}({thread_id}) | 시간: {start_time_str} | 질문: {query_data['query'][:30]}...")
        
        try:
            # 스레드 내에서는 timer_callback을 호출하지 않음 (Streamlit 컨텍스트 문제 방지)
            langflow_start = time.time()
            result = collect_query_data(
                query_data,
                langflow_url,
                langflow_api_key,
                temp_dir,
                session_id=session_id,
                timer_callback=None,  # 스레드 내에서는 None으로 설정
                query_index=idx
            )
            langflow_elapsed = time.time() - langflow_start
            
            # 소요 시간 계산
            query_elapsed_time = time.time() - query_start_time
            query_time_str = format_time(query_elapsed_time)
            result['elapsed_time'] = query_elapsed_time
            result['elapsed_time_str'] = query_time_str
            result['session_id'] = session_id
            result['query_index'] = idx  # 정렬을 위한 인덱스 추가
            
            # 병렬 처리 확인을 위한 로깅 (디버깅용)
            end_time_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[DEBUG] 쿼리 {idx+1} 완료 | 스레드: {thread_name}({thread_id}) | 시간: {end_time_str} | 총 소요: {query_time_str} | Langflow: {langflow_elapsed:.2f}초")
            
            return (idx, result)
        except Exception as e:
            # NoSessionContext 예외 특별 처리
            error_type_name = type(e).__name__
            
            # 예외 메시지 추출 (비어있을 경우 대체 메시지 제공)
            error_msg = str(e) if str(e) else repr(e)
            if not error_msg or error_msg == '':
                if error_type_name == 'NoSessionContext':
                    error_msg = "Streamlit 컨텍스트 오류: 스레드 내에서 Streamlit 함수를 호출할 수 없습니다. 이는 정상적인 동작이며 무시해도 됩니다."
                else:
                    error_msg = f"{error_type_name} occurred but no error message available"
            
            # traceback 정보도 포함 (디버깅용)
            import traceback
            try:
                tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
                # 너무 긴 경우 요약
                if len(tb_str) > 1000:
                    tb_str = tb_str[:1000] + "... (truncated)"
            except:
                tb_str = "Traceback 정보를 가져올 수 없습니다."
            
            # NoSessionContext는 실제 오류가 아니므로 성공으로 처리하거나 재시도
            if error_type_name == 'NoSessionContext':
                # 실제 작업을 다시 시도 (Streamlit 컨텍스트 없이)
                try:
                    # 환경변수만 설정하고 다시 시도
                    os.environ['LANGFLOW_URL'] = langflow_url
                    os.environ['LANGFLOW_API_KEY'] = langflow_api_key
                    
                    payload = {
                        "input_value": query_data["query"],
                        "output_type": "chat",
                        "input_type": "chat"
                    }
                    if session_id:
                        payload["session_id"] = session_id
                    
                    # ask_to_langflow 재시도
                    response_data, saved_response_filepath = ask_to_langflow(payload, url=langflow_url, return_filepath=True)
                    
                    saved_response_path = Path(saved_response_filepath)
                    if saved_response_path.exists():
                        with open(saved_response_path, 'r', encoding='utf-8') as f:
                            saved_response_data = json.load(f)
                    else:
                        saved_response_data = {'response': response_data}
                    
                    parsed_data = parse_response(saved_response_data)
                    parsed_path = temp_dir / "parsed_responses" / f"parsed_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}.json"
                    parsed_path.parent.mkdir(parents=True, exist_ok=True)
                    save_parsed_response(parsed_data, str(parsed_path))
                    
                    query_elapsed_time = time.time() - query_start_time
                    query_time_str = format_time(query_elapsed_time)
                    
                    return (idx, {
                        'query': query_data["query"],
                        'success': True,
                        'query_data': query_data,
                        'parsed_data': parsed_data,
                        'parsed_path': str(parsed_path),
                        'response_path': str(saved_response_path),
                        'session_id': session_id,
                        'elapsed_time': query_elapsed_time,
                        'elapsed_time_str': query_time_str,
                        'query_index': idx
                    })
                except Exception as retry_e:
                    # 재시도도 실패한 경우 원래 오류 사용
                    error_msg = f"재시도 실패: {str(retry_e) if str(retry_e) else repr(retry_e)}"
            
            error_result = {
                'query': query_data.get("query", "N/A"),
                'success': False,
                'error': error_msg,
                'error_type': error_type_name,
                'error_traceback': tb_str,
                'session_id': session_id,
                'elapsed_time': 0,
                'elapsed_time_str': '0초',
                'query_index': idx
            }
            return (idx, error_result)
    
    # 배치 처리 실행
    batch_start_time = time.time()
    print(f"[DEBUG] 배치 처리 시작: {len(queries_data)}개 쿼리, 배치 크기: {batch_size}, 시작 시간: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # 모든 작업 제출 (병렬 실행)
        future_to_index = {
            executor.submit(process_single_query, (idx, query_data)): idx
            for idx, query_data in enumerate(queries_data)
        }
        
        print(f"[DEBUG] 모든 작업 제출 완료: {len(future_to_index)}개 작업")
        
        # 완료된 작업 처리
        completed_count = 0
        results_dict = {}  # 인덱스를 키로 사용하여 결과 저장
        
        for future in as_completed(future_to_index):
            try:
                idx, result = future.result()
                results_dict[idx] = result
                completed_count += 1
                
                # 진행 상황 업데이트
                if progress_callback:
                    progress_callback(completed_count, total_queries)
                
                # 타이머 업데이트
                if timer_callback:
                    timer_callback()
                
                # 결과는 메인 스레드에서 표시하도록 수집만 함 (스레드 내에서 Streamlit 함수 호출 제거)
                # 결과 표시는 메인 스레드에서 처리
                
            except Exception as e:
                # 예외 발생 시에도 결과 추가
                idx = future_to_index[future]
                error_result = {
                    'query': queries_data[idx].get("query", "N/A"),
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'session_id': f"Q_{eval_start_str}_{idx+1:03d}",
                    'elapsed_time': 0,
                    'elapsed_time_str': '0초',
                    'query_index': idx
                }
                results_dict[idx] = error_result
                completed_count += 1
                
                if progress_callback:
                    progress_callback(completed_count, total_queries)
    
    # 인덱스 순서대로 정렬 (원래 순서 유지)
    collected_data = [results_dict[i] for i in sorted(results_dict.keys())]
    
    # 배치 처리 완료 시간 및 성능 분석
    batch_elapsed_time = time.time() - batch_start_time
    batch_time_str = format_time(batch_elapsed_time)
    print(f"[DEBUG] 배치 처리 완료: 총 {batch_elapsed_time:.2f}초 ({batch_time_str})")
    
    # 각 쿼리별 소요 시간 분석
    if collected_data:
        elapsed_times = [item.get('elapsed_time', 0) for item in collected_data if item.get('success')]
        if elapsed_times:
            avg_time = sum(elapsed_times) / len(elapsed_times)
            max_time = max(elapsed_times)
            min_time = min(elapsed_times)
            print(f"[DEBUG] 쿼리별 소요 시간 - 평균: {avg_time:.2f}초, 최대: {max_time:.2f}초, 최소: {min_time:.2f}초")
            print(f"[DEBUG] 병렬 처리 효율: 순차 처리 예상 시간 {sum(elapsed_times):.2f}초 vs 실제 {batch_elapsed_time:.2f}초")
            if batch_elapsed_time < sum(elapsed_times) * 0.8:
                print(f"[DEBUG] ✅ 병렬 처리 정상 작동 (약 {sum(elapsed_times)/batch_elapsed_time:.1f}배 속도 향상)")
            else:
                print(f"[DEBUG] ⚠️ 병렬 처리 미흡 (순차 처리와 유사)")
    
    return collected_data


def prepare_batch_evaluation_data(
    collected_data: List[Dict]
) -> Tuple[Dataset, List[Dict], List[Dict]]:
    """
    수집된 모든 데이터를 RAGAS 평가용 Dataset으로 준비합니다.
    답변이 없거나 context가 null인 케이스는 제외합니다.
    
    Args:
        collected_data: 수집된 데이터 리스트 (각 항목은 query_data와 parsed_data 포함)
    
    Returns:
        tuple: (Dataset, 평가에 포함된 데이터 리스트, 제외된 데이터 리스트(이유 포함))
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    ground_truths_list = []
    included_items = []  # 평가에 포함된 항목들
    excluded_items = []  # 평가에서 제외된 항목들 (이유 포함)
    
    for item in collected_data:
        if not item['success']:
            continue
        
        parsed_data = item.get('parsed_data', {})
        answer = parsed_data.get('answer', '')
        context = parsed_data.get('context', '')
        
        # 평가에서 제외할 케이스 체크
        # 1. 답변이 없는 경우
        if not answer or answer.strip() == '':
            excluded_items.append({
                **item,
                'exclusion_reason': '답변이 없거나 빈 문자열임'
            })
            continue
        
        # 2. 답변은 있지만 context가 null이거나 빈 문자열인 경우
        if not context or context.strip() == '':
            excluded_items.append({
                **item,
                'exclusion_reason': '답변은 있지만 context가 없거나 빈 문자열임'
            })
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
    return dataset, included_items, excluded_items


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
            value=st.session_state.langflow_url,
            help="Langflow API URL (예: http://10.1.1.70:7860/api/v1/run/...)",
            type="default",
            key="langflow_url_input"
        )
        
        # 입력값이 변경되면 세션 상태 업데이트
        if langflow_url != st.session_state.langflow_url:
            st.session_state.langflow_url = langflow_url
        
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
        
        # 배치 처리 설정
        st.subheader("⚡ 배치 처리 설정")
        batch_size = st.slider(
            "동시 처리 개수",
            min_value=1,
            max_value=20,
            value=5,
            help="한 번에 동시에 처리할 쿼리 개수 (높을수록 빠르지만 서버 부하 증가)"
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
                st.session_state.start_time = time.time()  # 시작 시간 기록
                
                # 평가 시작 시점의 날짜/시간 기록 (세션 ID 생성용)
                eval_start_datetime = datetime.now()
                eval_start_str = eval_start_datetime.strftime('%Y%m%d%H%M')  # YYYYMMDDHHMM 형식
                
                # 임시 디렉토리 생성
                temp_dir = Path(tempfile.mkdtemp(prefix="ragas_eval_"))
                zip_path = None  # ZIP 파일 경로 추적용
                
                try:
                    # 진행도 표시 영역
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    timer_text = st.empty()  # 타이머 표시 영역
                    results_container = st.container()
                    
                    # 경과 시간 계산 함수
                    def get_elapsed_time():
                        if 'start_time' in st.session_state:
                            elapsed = time.time() - st.session_state.start_time
                            return format_time(elapsed)
                        return "0초"
                    
                    # 시간 포맷팅 함수 (초 단위 입력)
                    def format_time(seconds):
                        """초 단위 시간을 읽기 쉬운 형식으로 변환"""
                        if seconds < 0:
                            return "0초"
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        millisecs = int((seconds % 1) * 1000)
                        
                        if hours > 0:
                            return f"{hours}시간 {minutes}분 {secs}초"
                        elif minutes > 0:
                            return f"{minutes}분 {secs}초"
                        elif secs > 0:
                            return f"{secs}.{millisecs//100}초"
                        else:
                            return f"{millisecs}ms"
                    
                    # 환경변수 설정
                    os.environ['LANGFLOW_URL'] = langflow_url
                    os.environ['LANGFLOW_API_KEY'] = langflow_api_key
                    os.environ['OPENAI_API_KEY'] = openai_api_key
                    
                    # 1단계: 모든 쿼리에 대해 배치로 데이터 수집
                    status_text.text(f"📥 데이터 수집 중... (배치 크기: {batch_size})")
                    timer_text.markdown(f"⏱️ **경과 시간:** {get_elapsed_time()}")
                    
                    # 타이머 업데이트 함수
                    def update_timer():
                        timer_text.markdown(f"⏱️ **경과 시간:** {get_elapsed_time()}")
                    
                    # 진행 상황 업데이트 함수
                    def update_progress(completed: int, total: int):
                        if not st.session_state.is_running:
                            return
                        progress = (completed / total) * 0.7  # 데이터 수집 단계: 0~70%
                        progress_bar.progress(progress)
                        status_text.text(f"📥 데이터 수집 중: {completed}/{total} 완료 (배치 크기: {batch_size})")
                    
                    # 배치 처리로 데이터 수집
                    # 성능 측정을 위한 시작 시간 기록
                    collection_start_time = time.time()
                    
                    collected_data = collect_query_data_batch(
                        queries_data=queries_data,
                        langflow_url=langflow_url,
                        langflow_api_key=langflow_api_key,
                        temp_dir=temp_dir,
                        eval_start_str=eval_start_str,
                        batch_size=batch_size,
                        timer_callback=update_timer,
                        progress_callback=update_progress,
                        results_container=None  # 스레드 내 Streamlit 호출 방지를 위해 None으로 설정
                    )
                    
                    # 데이터 수집 성능 분석
                    collection_elapsed = time.time() - collection_start_time
                    successful_count = sum(1 for item in collected_data if item.get('success'))
                    
                    # 성능 정보 표시
                    if successful_count > 0:
                        avg_query_time = sum(item.get('elapsed_time', 0) for item in collected_data if item.get('success')) / successful_count
                        st.info(f"📊 데이터 수집 완료: {successful_count}개 성공, 총 {collection_elapsed:.2f}초, 평균 {avg_query_time:.2f}초/질문")
                        
                        # 병렬 처리 효율성 확인
                        total_sequential_time = sum(item.get('elapsed_time', 0) for item in collected_data if item.get('success'))
                        if total_sequential_time > 0:
                            speedup = total_sequential_time / collection_elapsed if collection_elapsed > 0 else 1
                            if speedup > 1.5:
                                st.success(f"✅ 병렬 처리 정상 작동 (약 {speedup:.1f}배 속도 향상)")
                            elif speedup > 1.1:
                                st.warning(f"⚠️ 병렬 처리 부분 작동 (약 {speedup:.1f}배 속도 향상, 개선 필요)")
                            else:
                                st.error(f"❌ 병렬 처리 미작동 (순차 처리와 유사, {speedup:.1f}배)")
                    
                    # 중지 확인
                    if not st.session_state.is_running:
                        st.warning("⚠️ 평가가 중지되었습니다.")
                        st.stop()
                    
                    # 배치 처리 완료 후 결과 표시 (메인 스레드에서)
                    with results_container:
                        st.markdown("### 📋 데이터 수집 결과")
                        for idx, result in enumerate(collected_data, 1):
                            session_id_display = result.get('session_id', 'N/A')
                            query_time_str = result.get('elapsed_time_str', '0초')
                            if result['success']:
                                st.success(f"✅ [{idx}/{len(collected_data)}] {result['query'][:50]}... (세션: {session_id_display}, 소요 시간: {query_time_str})")
                            else:
                                error_msg = result.get('error', 'Unknown error')
                                error_type = result.get('error_type', 'Unknown')
                                error_traceback = result.get('error_traceback', '')
                                st.error(f"❌ [{idx}/{len(collected_data)}] {result['query'][:50]}... (세션: {session_id_display}, 소요 시간: {query_time_str})")
                                with st.expander(f"오류 상세 정보 (쿼리 {idx})"):
                                    st.code(f"오류 타입: {error_type}\n오류 메시지: {error_msg}")
                                    if error_traceback:
                                        st.code(f"Traceback:\n{error_traceback}", language='python')
                    
                    # 2단계: 모든 데이터 수집 완료 후 평가 실행
                    if st.session_state.is_running:
                        status_text.text("📊 평가 실행 중...")
                        update_timer()  # 타이머 업데이트
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
                                dataset, included_items, excluded_items = prepare_batch_evaluation_data(successful_data)
                                
                                # 제외된 케이스 확인 및 구체적인 이유 표시
                                if excluded_items:
                                    # 제외 이유별로 그룹화
                                    excluded_by_reason = {}
                                    for item in excluded_items:
                                        reason = item.get('exclusion_reason', '알 수 없는 이유')
                                        if reason not in excluded_by_reason:
                                            excluded_by_reason[reason] = []
                                        excluded_by_reason[reason].append(item)
                                    
                                    st.warning(f"⚠️ 다음 {len(excluded_items)}개 쿼리는 평가에서 제외되었습니다:")
                                    for reason, items in excluded_by_reason.items():
                                        st.caption(f"**{reason}** ({len(items)}개):")
                                        for item in items:
                                            st.caption(f"  - {item['query'][:60]}...")
                                
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
                                
                                # LLM 인스턴스 생성 (max_tokens를 명시적으로 설정)
                                # llm_factory에 max_tokens를 직접 전달하거나, 생성 후 model_args에 설정
                                llm = llm_factory("gpt-4o-mini", client=openai_client, max_tokens=4096)
                                
                                # max_tokens를 4096으로 명시적으로 설정 (model_args에 직접 설정)
                                if hasattr(llm, 'model_args') and isinstance(llm.model_args, dict):
                                    llm.model_args['max_tokens'] = 4096
                                    print(f"✅ max_tokens를 4096으로 명시적으로 설정했습니다 (model_args: {llm.model_args.get('max_tokens')})")
                                else:
                                    # model_args가 없는 경우 대비
                                    try:
                                        setattr(llm, 'max_tokens', 4096)
                                        print("✅ max_tokens를 4096으로 설정했습니다 (동적 속성)")
                                    except Exception as e:
                                        print(f"⚠️ 경고: max_tokens를 4096으로 설정할 수 없습니다. (오류: {e})")
                                
                                # 평가 메트릭 설정
                                # answer_relevancy는 기본적으로 3개의 질문을 생성하여 평가합니다
                                # generations 파라미터로 조정 가능 (기본값: 3)
                                try:
                                    # answer_relevancy에 generations 파라미터 설정 시도
                                    metrics = [
                                        faithfulness.__class__(llm=llm),
                                        answer_relevancy.__class__(llm=llm, generations=3),  # 명시적으로 3개 설정
                                        context_precision.__class__(llm=llm),
                                        context_recall.__class__(llm=llm),
                                    ]
                                except TypeError:
                                    # generations 파라미터를 지원하지 않는 버전인 경우
                                    metrics = [
                                        faithfulness.__class__(llm=llm),
                                        answer_relevancy.__class__(llm=llm),
                                        context_precision.__class__(llm=llm),
                                        context_recall.__class__(llm=llm),
                                    ]
                                
                                # 전체 평가 실행 (RAGAS는 이미 배치로 처리하지만, 성능 측정 및 병렬 처리 확인)
                                progress_bar.progress(0.9)
                                update_timer()  # 평가 시작 전 타이머 업데이트
                                
                                # 평가 시작 시간 기록
                                eval_start_time = time.time()
                                eval_start_time_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                
                                print(f"[DEBUG] 평가 단계 시작: {len(dataset)}개 항목, {len(metrics)}개 메트릭, 시작 시간: {eval_start_time_str}")
                                print(f"[DEBUG] 평가 메트릭: {[m.__class__.__name__ for m in metrics]}")
                                
                                # RAGAS evaluate는 이미 Dataset 전체를 배치로 처리합니다
                                # 하지만 내부적으로 각 메트릭을 순차적으로 평가할 수 있으므로
                                # 성능 측정을 추가하여 실제 처리 시간 확인
                                st.info(f"📊 {len(dataset)}개 항목에 대해 {len(metrics)}개 메트릭 평가 중... (RAGAS 배치 처리)")
                                
                                # 각 메트릭별 평가 시간 측정을 위한 로깅
                                print(f"[DEBUG] RAGAS evaluate 함수 호출 시작...")
                                metric_start_time = time.time()
                                
                                full_result = evaluate(dataset=dataset, metrics=metrics)
                                
                                metric_end_time = time.time()
                                metric_duration = metric_end_time - metric_start_time
                                
                                results_df = full_result.to_pandas()
                                
                                # 평가 완료 시간 기록
                                eval_end_time = time.time()
                                eval_duration = eval_end_time - eval_start_time
                                eval_end_time_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                
                                # 성능 정보 표시 및 로깅
                                avg_time_per_item = eval_duration / len(dataset) if len(dataset) > 0 else 0
                                total_expected_time = avg_time_per_item * len(dataset) * len(metrics)  # 순차 처리 예상 시간
                                
                                print(f"[DEBUG] 평가 단계 완료: 총 {eval_duration:.2f}초, 종료 시간: {eval_end_time_str}")
                                print(f"[DEBUG] 평가 성능 분석:")
                                print(f"  - 총 소요 시간: {eval_duration:.2f}초")
                                print(f"  - 항목당 평균: {avg_time_per_item:.2f}초")
                                print(f"  - 메트릭당 평균: {eval_duration / len(metrics):.2f}초 (총 {len(metrics)}개 메트릭)")
                                print(f"  - 항목×메트릭 조합: {len(dataset)}개 항목 × {len(metrics)}개 메트릭 = {len(dataset) * len(metrics)}개 평가")
                                print(f"  - 조합당 평균: {eval_duration / (len(dataset) * len(metrics)):.2f}초")
                                
                                # 순차 처리 예상 시간과 비교
                                if total_expected_time > 0:
                                    speedup_ratio = total_expected_time / eval_duration if eval_duration > 0 else 1
                                    print(f"  - 순차 처리 예상: {total_expected_time:.2f}초 vs 실제: {eval_duration:.2f}초")
                                    if speedup_ratio > 1.5:
                                        print(f"  - ✅ 병렬 처리 가능성 있음 (약 {speedup_ratio:.1f}배)")
                                    elif speedup_ratio > 1.1:
                                        print(f"  - ⚠️ 부분 병렬 처리 (약 {speedup_ratio:.1f}배)")
                                    else:
                                        print(f"  - ❌ 순차 처리로 보임 ({speedup_ratio:.1f}배)")
                                
                                st.success(f"✅ 평가 완료: 총 {eval_duration:.2f}초 ({len(dataset)}개 항목, 항목당 평균 {avg_time_per_item:.2f}초)")
                                
                                update_timer()  # 평가 완료 후 타이머 업데이트
                                
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
                                        'parsed_path': item.get('parsed_path'),
                                        'elapsed_time': item.get('elapsed_time', 0),
                                        'elapsed_time_str': item.get('elapsed_time_str', '0초')
                                    })
                                
                                # 평가에서 제외된 항목들 (구체적인 제외 이유 포함)
                                for item in excluded_items:
                                    exclusion_reason = item.get('exclusion_reason', '알 수 없는 이유')
                                    evaluation_results_list.append({
                                        'query': item['query'],
                                        'success': False,
                                        'error': f'{exclusion_reason}으로 인해 평가에서 제외됨',
                                        'error_type': 'ExcludedFromEvaluation',
                                        'exclusion_reason': exclusion_reason,
                                        'elapsed_time': item.get('elapsed_time', 0),
                                        'elapsed_time_str': item.get('elapsed_time_str', '0초')
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
                                total_time = get_elapsed_time()
                                status_text.text("✅ 모든 평가가 완료되었습니다!")
                                timer_text.markdown(f"✅ **총 소요 시간:** {total_time}")
                            
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
                        total_time = get_elapsed_time()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("전체 쿼리", len(st.session_state.evaluation_results))
                        with col2:
                            st.metric("성공", successful, delta=f"{successful/len(st.session_state.evaluation_results)*100:.1f}%")
                        with col3:
                            st.metric("실패", failed, delta=f"-{failed/len(st.session_state.evaluation_results)*100:.1f}%")
                        with col4:
                            st.metric("총 소요 시간", total_time)
                        
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
                                elapsed_time_str = result.get('elapsed_time_str', '측정 안 됨')
                                if result['success']:
                                    st.markdown(f"#### {i}. {result['query']} ⏱️ {elapsed_time_str}")
                                    st.json(result['results'])
                                else:
                                    st.markdown(f"#### {i}. {result['query']} ❌ ⏱️ {elapsed_time_str}")
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

