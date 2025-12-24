"""
askToLangflow 모듈
URL로 POST 요청을 보내고 응답을 저장하는 기능을 제공합니다.
"""

import json
import os
import re
import sys
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Windows 콘솔에서 UTF-8 출력 지원
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6 이하 버전에서는 reconfigure가 없음
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# .env 파일 로드
load_dotenv()


def decode_unicode_escape(obj):
    """
    딕셔너리나 리스트 내의 모든 유니코드 이스케이프 문자열을 한국어로 디코딩합니다.
    
    Args:
        obj: 딕셔너리, 리스트, 또는 문자열
    
    Returns:
        디코딩된 객체
    """
    if isinstance(obj, dict):
        return {key: decode_unicode_escape(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [decode_unicode_escape(item) for item in obj]
    elif isinstance(obj, str):
        # 유니코드 이스케이프 시퀀스(\uXXXX)를 실제 문자로 변환
        try:
            # JSON 문자열로 감싸서 파싱하면 유니코드 이스케이프가 자동으로 디코딩됨
            return json.loads(f'"{obj}"')
        except (json.JSONDecodeError, ValueError):
            # 파싱 실패 시 원본 반환
            return obj
    else:
        return obj


def parse_streaming_response(text: str) -> dict:
    """
    스트리밍 응답 텍스트를 파싱하여 한국어로 디코딩된 딕셔너리로 반환합니다.
    
    Args:
        text: 스트리밍 응답 텍스트 (여러 JSON 객체가 줄바꿈으로 구분됨)
    
    Returns:
        파싱 및 디코딩된 응답 데이터
    """
    if not text:
        return {'text': ''}
    
    # 줄바꿈으로 분리하여 각 JSON 객체 파싱
    lines = text.strip().split('\n')
    parsed_events = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            # JSON 파싱
            event_data = json.loads(line)
            # 유니코드 이스케이프 디코딩
            decoded_data = decode_unicode_escape(event_data)
            parsed_events.append(decoded_data)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 원본 텍스트 유지
            continue
    
    # 파싱된 이벤트들을 구조화
    if parsed_events:
        # 마지막 'end' 이벤트에서 최종 결과 추출
        for event in reversed(parsed_events):
            if isinstance(event, dict) and event.get('event') == 'end':
                result = event.get('data', {}).get('result', {})
                if result:
                    return decode_unicode_escape(result)
        
        # 'end' 이벤트가 없으면 모든 이벤트 반환
        return {'events': parsed_events, 'text': text}
    
    # 파싱 실패 시 원본 텍스트 반환 (유니코드 디코딩 시도)
    try:
        decoded_text = decode_unicode_escape(text)
        return {'text': decoded_text if isinstance(decoded_text, str) else text}
    except:
        return {'text': text}


def ask_to_langflow(payload: dict, url: str = None, headers: dict = None) -> dict:
    """
    지정된 URL로 POST 요청을 보내고 응답을 저장합니다.
    
    Args:
        payload (dict): POST 요청에 포함할 데이터
        url (str, optional): POST 요청을 보낼 URL. 기본값은 None (환경변수 LANGFLOW_URL에서 가져옴)
        headers (dict, optional): 요청 헤더. 기본값은 None (Content-Type: application/json 사용)
    
    Returns:
        dict: 서버로부터 받은 응답 데이터
    
    Raises:
        ValueError: URL이 제공되지 않고 환경변수에도 없을 때
        requests.RequestException: HTTP 요청 중 오류 발생 시
    """
    # URL이 제공되지 않으면 환경변수에서 가져오기
    if url is None:
        url = os.getenv('LANGFLOW_URL')
        if url is None:
            raise ValueError("URL이 제공되지 않았고 환경변수 LANGFLOW_URL도 설정되지 않았습니다.")
    
    # 기본 헤더 설정
    if headers is None:
        headers = {
            'Content-Type': 'application/json'
        }
    
    # x-api-key 헤더 추가 (환경변수에서 가져오기)
    api_key = os.getenv('LANGFLOW_API_KEY')
    if api_key:
        headers['x-api-key'] = api_key
    
    # POST 요청 전송
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        
        # 응답 데이터 추출 및 한국어 디코딩
        content_type = response.headers.get('content-type', '')
        if content_type.startswith('application/json'):
            try:
                response_data = response.json()
                # 유니코드 이스케이프 디코딩
                response_data = decode_unicode_escape(response_data)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트로 처리
                response_data = parse_streaming_response(response.text)
        else:
            # 텍스트 응답인 경우 스트리밍 응답 파싱
            response_data = parse_streaming_response(response.text)
        
        # 응답 저장
        save_response(url, payload, response_data, response.status_code)
        
        return response_data
    
    except requests.exceptions.RequestException as e:
        # 오류 발생 시에도 로그 저장
        error_data = {
            'error': str(e),
            'error_type': type(e).__name__
        }
        save_response(url, payload, error_data, status_code=None, is_error=True)
        raise


def save_payload(payload: dict, filename: str = None) -> str:
    """
    payload를 JSON 파일로 저장합니다.
    
    Args:
        payload (dict): 저장할 payload 데이터
        filename (str, optional): 저장할 파일명. 기본값은 None (타임스탬프 기반 자동 생성)
    
    Returns:
        str: 저장된 파일 경로
    """
    # 저장 디렉토리 경로 설정
    save_dir = Path('data/payloads')
    
    # 파일명이 지정되지 않으면 타임스탬프 기반으로 생성
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"payload_{timestamp}.json"
    
    # .json 확장자가 없으면 추가
    if not filename.endswith('.json'):
        filename += '.json'
    
    filepath = save_dir / filename
    
    # JSON 파일로 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    print(f"Payload가 저장되었습니다: {filepath}")
    return str(filepath)


def load_payload(filepath: str) -> dict:
    """
    저장된 payload 파일을 불러옵니다.
    
    Args:
        filepath (str): 불러올 payload 파일 경로
    
    Returns:
        dict: 불러온 payload 데이터
    
    Raises:
        FileNotFoundError: 파일이 존재하지 않을 때
        json.JSONDecodeError: JSON 파싱 오류 발생 시
    """
    filepath = Path(filepath)
    
    # 파일이 존재하는지 확인
    if not filepath.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    # JSON 파일 읽기
    with open(filepath, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    
    print(f"Payload를 불러왔습니다: {filepath}")
    return payload


def save_response(url: str, payload: dict, response_data: dict, status_code: int = None, is_error: bool = False):
    """
    응답 데이터를 파일로 저장합니다.
    
    Args:
        url (str): 요청한 URL
        payload (dict): 전송한 payload
        response_data (dict): 받은 응답 데이터
        status_code (int, optional): HTTP 상태 코드
        is_error (bool): 오류 발생 여부
    """
    # 저장 디렉토리 경로 설정
    save_dir = Path('data/responses')
    
    # 파일명 생성 (타임스탬프 기반)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"response_{timestamp}.json"
    filepath = save_dir / filename
    
    # 저장할 데이터 구성
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'url': url,
        'payload': payload,
        'status_code': status_code,
        'is_error': is_error,
        'response': response_data
    }
    
    # JSON 파일로 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"응답이 저장되었습니다: {filepath}")


if __name__ == '__main__':
    # Payload 불러오기 예시
    print("=== Payload 불러오기 예시 ===")
    loaded_payload = load_payload("data/payloads/payload.json")
    print(f"불러온 payload: {loaded_payload}")
    
    # POST 요청 예시 (환경변수에서 URL 가져오기)
    print("\n=== POST 요청 예시 ===")
    try:
        result = ask_to_langflow(loaded_payload)
        print("요청 성공!")
        # 한국어로 출력되도록 JSON 포맷팅 (인코딩 오류 방지)
        try:
            response_str = json.dumps(result, ensure_ascii=False, indent=2)
            print(f"응답: {response_str}")
        except UnicodeEncodeError:
            # 인코딩 오류 발생 시 파일로 저장하고 경로만 출력
            print("응답이 너무 크거나 특수 문자가 포함되어 있습니다.")
            print("저장된 파일을 확인하세요.")
    except ValueError as e:
        print(f"설정 오류: {e}")
        print("환경변수 LANGFLOW_URL과 LANGFLOW_API_KEY를 설정하거나 .env 파일을 확인하세요.")
    except Exception as e:
        print(f"요청 실패: {e}")

