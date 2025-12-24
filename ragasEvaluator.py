"""
ragasEvaluator 모듈
RAGAS를 사용하여 RAG 시스템을 평가하고 결과를 저장하는 기능을 제공합니다.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# .env 파일 로드
load_dotenv()

# Windows 콘솔에서 UTF-8 출력 지원
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def load_ragas_data(filepath: str = "data/ragas_data/data.json") -> List[Dict]:
    """
    RAGAS 평가용 데이터를 불러옵니다.

    Args:
        filepath (str): 데이터 파일 경로

    Returns:
        List[Dict]: 평가 데이터 리스트
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"RAGAS 데이터를 불러왔습니다: {filepath} ({len(data)}개 항목)")
    return data


def load_parsed_response(filepath: str) -> Dict:
    """
    파싱된 응답 파일을 불러옵니다.

    Args:
        filepath (str): 파싱된 응답 파일 경로

    Returns:
        Dict: 파싱된 응답 데이터
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def prepare_evaluation_data(
    ragas_data: List[Dict],
    parsed_response: Dict
) -> Dataset:
    """
    RAGAS 평가를 위한 데이터를 준비합니다.

    Args:
        ragas_data (List[Dict]): RAGAS 평가 데이터
        parsed_response (Dict): 파싱된 응답 데이터

    Returns:
        Dataset: 평가용 Dataset
    """
    # 첫 번째 항목 사용 (나중에 여러 항목 지원 가능)
    if len(ragas_data) == 0:
        raise ValueError("RAGAS 데이터가 비어있습니다.")
    
    eval_data = ragas_data[0]
    
    # 데이터 딕셔너리 생성
    data_dict = {
        "question": [eval_data["query"]],
        "answer": [parsed_response["answer"]],
        "contexts": [[parsed_response["context"]]],
        "ground_truth": [eval_data["ground_truth"]],
        "ground_truths": [[eval_data["gt_context"]]],
    }
    
    # Dataset 생성
    dataset = Dataset.from_dict(data_dict)
    
    return dataset


def run_ragas_evaluation(dataset: Dataset) -> Dict:
    """
    RAGAS 평가를 실행합니다.

    Args:
        dataset (Dataset): 평가용 Dataset

    Returns:
        Dict: 평가 결과
    """
    print("\n=== RAGAS 평가 시작 ===")
    
    # OpenAI API 키 확인
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
            ".env 파일에 OPENAI_API_KEY를 추가하거나 환경변수를 설정하세요."
        )
    
    # OpenAI 클라이언트 생성
    openai_client = OpenAI(api_key=api_key)
    
    # LLM 인스턴스 생성 (llm_factory 사용)
    llm = llm_factory("gpt-4o-mini", client=openai_client)
    
    # Embeddings 인스턴스 생성 (OpenAIEmbeddings 직접 사용)
    embeddings = OpenAIEmbeddings(
        client=openai_client,
        model="text-embedding-ada-002"
    )
    
    # 평가 메트릭 정의
    # ragas.metrics에서 import한 것은 인스턴스이므로 클래스를 사용하여 새 인스턴스 생성
    # answer_relevancy는 embeddings 인터페이스 호환성 문제로 인해 embeddings 없이 사용
    # (embeddings가 없어도 llm만으로 평가 가능하지만 정확도가 낮을 수 있음)
    metrics = [
        faithfulness.__class__(llm=llm),
        answer_relevancy.__class__(llm=llm),  # embeddings 제거 - 인터페이스 호환성 문제
        context_precision.__class__(llm=llm),
        context_recall.__class__(llm=llm),
    ]
    
    # 평가 실행
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
    )
    
    # 결과를 딕셔너리로 변환
    result_df = result.to_pandas()
    result_dict = result_df.to_dict('records')[0]
    
    print("평가 완료!")
    print(f"\n평가 결과:")
    for metric, score in result_dict.items():
        if isinstance(score, (int, float)):
            print(f"  {metric}: {score:.4f}")
    
    return result_dict


def save_evaluation_results(
    results: Dict,
    ragas_data: List[Dict],
    parsed_response: Dict,
    output_dir: str = "results"
) -> tuple:
    """
    평가 결과를 JSON과 JPG 이미지로 저장합니다.

    Args:
        results (Dict): 평가 결과
        ragas_data (List[Dict]): 원본 RAGAS 데이터
        parsed_response (Dict): 파싱된 응답 데이터
        output_dir (str): 저장 디렉토리

    Returns:
        tuple: (JSON 파일 경로, JPG 파일 경로)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # JSON 저장
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_results": results,
        "input_data": {
            "query": ragas_data[0]["query"] if ragas_data else None,
            "ground_truth": ragas_data[0]["ground_truth"] if ragas_data else None,
            "gt_context": ragas_data[0]["gt_context"] if ragas_data else None,
        },
        "response_data": {
            "answer": parsed_response["answer"],
            "context": parsed_response["context"],
        }
    }
    
    json_filepath = output_dir / f"ragas_results_{timestamp}.json"
    with open(json_filepath, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n평가 결과가 저장되었습니다: {json_filepath}")
    
    # JPG 이미지 저장
    jpg_filepath = output_dir / f"ragas_results_{timestamp}.jpg"
    create_evaluation_visualization(results, jpg_filepath)
    
    print(f"평가 시각화가 저장되었습니다: {jpg_filepath}")
    
    return str(json_filepath), str(jpg_filepath)


def create_evaluation_visualization(results: Dict, filepath: Path):
    """
    평가 결과를 시각화하여 이미지로 저장합니다.

    Args:
        results (Dict): 평가 결과
        filepath (Path): 저장할 파일 경로
    """
    # 메트릭 이름과 점수 추출
    metrics = []
    scores = []
    
    for key, value in results.items():
        if isinstance(value, (int, float)) and key not in ['question', 'answer', 'contexts', 'ground_truth', 'ground_truths']:
            metrics.append(key)
            scores.append(value)
    
    if not metrics:
        print("시각화할 메트릭이 없습니다.")
        return
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(metrics, scores, color=colors[:len(metrics)])
    
    # 값 표시
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('점수', fontsize=12)
    ax.set_xlabel('메트릭', fontsize=12)
    ax.set_title('RAGAS 평가 결과', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"시각화 이미지가 저장되었습니다: {filepath}")


def evaluate_rag_system(
    ragas_data_path: str = "data/ragas_data/data.json",
    parsed_response_path: str = "data/parsed_responses/parsed_20251224_182902_039576.json",
    output_dir: str = "results"
) -> tuple:
    """
    RAG 시스템을 평가하고 결과를 저장합니다.

    Args:
        ragas_data_path (str): RAGAS 평가 데이터 파일 경로
        parsed_response_path (str): 파싱된 응답 파일 경로
        output_dir (str): 결과 저장 디렉토리

    Returns:
        tuple: (JSON 파일 경로, JPG 파일 경로)
    """
    # 데이터 로드
    ragas_data = load_ragas_data(ragas_data_path)
    parsed_response = load_parsed_response(parsed_response_path)
    
    # 평가 데이터 준비
    dataset = prepare_evaluation_data(ragas_data, parsed_response)
    
    # 평가 실행
    results = run_ragas_evaluation(dataset)
    
    # 결과 저장
    json_path, jpg_path = save_evaluation_results(
        results, ragas_data, parsed_response, output_dir
    )
    
    return json_path, jpg_path


if __name__ == '__main__':
    print("=== RAGAS 평가 모듈 실행 ===")
    try:
        json_path, jpg_path = evaluate_rag_system()
        print(f"\n✅ 평가 완료!")
        print(f"   JSON: {json_path}")
        print(f"   JPG: {jpg_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

