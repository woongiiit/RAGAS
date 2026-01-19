"""
responseParser 모듈
Langflow 응답에서 실제 답변과 참고한 context를 추출하는 기능을 제공합니다.
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _extract_content_from_chroma_result(context_raw: str) -> str:
    """
    chromaResult 형태의 문자열에서 content 값만 추출합니다.
    
    Args:
        context_raw (str): chromaResult를 포함한 문자열
    
    Returns:
        str: 추출된 content 값들을 합친 문자열
    """
    try:
        # 작은따옴표를 큰따옴표로 변환하여 JSON으로 파싱 시도
        # 단, 키 이름의 작은따옴표는 유지하고 값의 작은따옴표만 변환
        json_str = context_raw.replace("'", '"')
        context_dict = json.loads(json_str)
        
        # chromaResult에서 content 추출
        if isinstance(context_dict, dict) and 'chromaResult' in context_dict:
            chroma_results = context_dict['chromaResult']
            if isinstance(chroma_results, list):
                # 각 항목의 content를 추출하여 리스트로 만든 후 합침
                contents = []
                for item in chroma_results:
                    if isinstance(item, dict) and 'content' in item:
                        contents.append(item['content'])
                # 여러 content를 줄바꿈으로 구분하여 합침
                return '\n\n'.join(contents) if contents else context_raw
    except (json.JSONDecodeError, ValueError):
        # JSON 파싱 실패 시 정규표현식으로 시도
        contents = []
        # 'content': '...' 패턴 찾기 (간단한 버전)
        pattern = r"'content':\s*'([^']*(?:'[^',}\]]*)*)'"
        matches = re.finditer(pattern, context_raw)
        
        for match in matches:
            content = match.group(1)
            # 이스케이프 문자 처리
            content = content.replace("\\'", "'").replace("\\\\", "\\")
            contents.append(content)
        
        if contents:
            return '\n\n'.join(contents)
    
    # 추출 실패 시 원본 반환
    return context_raw


def parse_response(response_data: dict) -> Dict[str, Optional[str]]:
    """
    Langflow 응답에서 실제 답변과 참고한 context를 추출합니다.

    Args:
        response_data (dict): Langflow 응답 데이터 (전체 응답 또는 response 필드만)

    Returns:
        dict: {
            'answer': 실제 사용자에게 보여지는 답변,
            'context': 답변 생성 시 모델이 참고한 context
        }
    """
    # response 필드가 있으면 그것을 사용, 없으면 전체를 response로 간주
    if 'response' in response_data:
        response = response_data['response']
    else:
        response = response_data

    answer = None
    context = None

    try:
        # 실제 답변 추출: response.outputs[0].outputs[0].results.message.data.text
        if 'outputs' in response and len(response['outputs']) > 0:
            first_output = response['outputs'][0]
            if 'outputs' in first_output and len(first_output['outputs']) > 0:
                second_output = first_output['outputs'][0]
                if 'results' in second_output and 'message' in second_output['results']:
                    message = second_output['results']['message']
                    if 'data' in message and 'text' in message['data']:
                        answer = message['data']['text']
                    elif 'text' in message:
                        answer = message['text']

        # Context 추출: content_blocks에서 type이 'json'인 항목들의 data.text 수집
        if 'outputs' in response and len(response['outputs']) > 0:
            first_output = response['outputs'][0]
            if 'outputs' in first_output and len(first_output['outputs']) > 0:
                second_output = first_output['outputs'][0]
                if 'results' in second_output and 'message' in second_output['results']:
                    message = second_output['results']['message']
                    
                    # content_blocks는 message.content_blocks 또는 message.data.content_blocks에 있을 수 있음
                    content_blocks = None
                    if 'content_blocks' in message:
                        content_blocks = message['content_blocks']
                    elif 'data' in message and 'content_blocks' in message['data']:
                        content_blocks = message['data']['content_blocks']
                    
                    if content_blocks and len(content_blocks) > 0:
                        # 모든 content_blocks를 순회하며 type이 'json'인 항목의 data.text 수집
                        text_contents = []
                        for content_block in content_blocks:
                            if 'contents' in content_block:
                                for content in content_block['contents']:
                                    if content.get('type') == 'json' and 'data' in content:
                                        data = content['data']
                                        if 'text' in data and data['text']:
                                            raw_text = data['text']
                                            # chromaResult 구조에서 content만 추출하여 정제
                                            extracted_content = _extract_content_from_chroma_result(raw_text)
                                            text_contents.append(extracted_content)
                        
                        # 수집한 text들을 줄바꿈으로 구분하여 합침
                        # 빈 문자열은 제외
                        text_contents = [tc for tc in text_contents if tc and tc.strip()]
                        if text_contents:
                            context = '\n\n'.join(text_contents)

    except (KeyError, IndexError, TypeError) as e:
        print(f"응답 파싱 중 오류 발생: {e}")
        return {'answer': None, 'context': None}

    return {
        'answer': answer,
        'context': context
    }


def load_and_parse_response(filepath: str) -> Dict[str, Optional[str]]:
    """
    저장된 응답 파일을 불러와서 파싱합니다.

    Args:
        filepath (str): 응답 JSON 파일 경로

    Returns:
        dict: {
            'answer': 실제 사용자에게 보여지는 답변,
            'context': 답변 생성 시 모델이 참고한 context
        }

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
        response_data = json.load(f)

    print(f"응답 파일을 불러왔습니다: {filepath}")
    return parse_response(response_data)


def save_parsed_response(parsed_data: dict, filepath: str = None) -> str:
    """
    파싱된 응답 데이터를 JSON 파일로 저장합니다.

    Args:
        parsed_data (dict): parse_response() 또는 load_and_parse_response()의 결과
        filepath (str, optional): 저장할 파일 경로. 기본값은 None (자동 생성)

    Returns:
        str: 저장된 파일 경로
    """
    from datetime import datetime

    # 저장 디렉토리 경로 설정
    save_dir = Path('data/parsed_responses')

    # 파일명이 지정되지 않으면 타임스탬프 + UUID 기반으로 생성
    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"parsed_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        filepath = save_dir / filename
    else:
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.json')

    # 디렉토리가 없으면 생성
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # JSON 파일로 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, ensure_ascii=False, indent=2)

    print(f"파싱된 응답이 저장되었습니다: {filepath}")
    return str(filepath)


if __name__ == '__main__':
    # 예시: 저장된 응답 파일 파싱
    print("=== 응답 파일 파싱 예시 ===")
    try:
        parsed = load_and_parse_response("data/responses/response_20251224_182839_008325.json")
        print("\n=== 파싱 결과 ===")
        print(f"답변 길이: {len(parsed['answer']) if parsed['answer'] else 0} 문자")
        print(f"Context 길이: {len(parsed['context']) if parsed['context'] else 0} 문자")
        print(f"\n답변 (처음 200자):")
        if parsed['answer']:
            print(parsed['answer'][:200] + "..." if len(parsed['answer']) > 200 else parsed['answer'])
        print(f"\nContext (처음 200자):")
        if parsed['context']:
            print(parsed['context'][:200] + "..." if len(parsed['context']) > 200 else parsed['context'])
        
        # 파싱된 결과 저장
        print("\n=== 파싱된 결과 저장 ===")
        save_parsed_response(parsed)
    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"오류 발생: {e}")

