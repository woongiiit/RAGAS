# RAGAS 평가 시스템

Langflow API를 사용한 RAG 시스템의 RAGAS 평가 도구입니다.

## 프로젝트 구조

```
RAGAS_V3/
├── askToLangflow.py      # Langflow API로 POST 요청 전송 및 응답 저장
├── responseParser.py     # 응답에서 답변과 context 추출
├── ragasEvaluator.py     # RAGAS를 사용한 RAG 시스템 평가
├── requirements.txt       # 프로젝트 의존성
├── .envExample           # 환경변수 예시 파일
├── data/
│   ├── payloads/         # 요청 payload 파일
│   ├── responses/        # Langflow 응답 파일
│   ├── parsed_responses/ # 파싱된 응답 파일
│   └── ragas_data/       # RAGAS 평가용 데이터
└── results/              # 평가 결과 (JSON, JPG)
```

## 주요 기능

### 1. askToLangflow 모듈
- Langflow API로 POST 요청 전송
- 응답을 `data/responses/` 폴더에 저장
- 한국어 유니코드 디코딩 지원
- Payload 파일 저장/불러오기 기능

### 2. responseParser 모듈
- Langflow 응답에서 실제 답변(`answer`) 추출
- 답변 생성 시 참고한 context 추출
- `chromaResult`에서 `content` 값만 추출
- 파싱된 결과를 `data/parsed_responses/` 폴더에 저장

### 3. ragasEvaluator 모듈
- RAGAS를 사용한 RAG 시스템 평가
- 평가 메트릭:
  - `faithfulness`: 답변이 context에 기반한 정도
  - `answer_relevancy`: 답변이 질문과 관련있는 정도
  - `context_precision`: context가 질문과 관련있는 정도
  - `context_recall`: ground truth context와 실제 context의 일치도
- 평가 결과를 JSON과 JPG 이미지로 저장

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 환경변수 설정:
```bash
cp .envExample .env
# .env 파일에 실제 값 입력
```

## 사용 방법

### 1. Langflow API 요청 전송
```bash
python askToLangflow.py
```

### 2. 응답 파싱
```bash
python responseParser.py
```

### 3. RAGAS 평가 실행
```bash
python ragasEvaluator.py
```

## 환경변수

`.env` 파일에 다음 변수를 설정해야 합니다:

- `LANGFLOW_URL`: Langflow API URL
- `LANGFLOW_API_KEY`: Langflow API Key
- `OPENAI_API_KEY`: OpenAI API Key (RAGAS 평가에 사용)

## 데이터 형식

### RAGAS 평가 데이터 (`data/ragas_data/data.json`)
```json
[
  {
    "query": "질문",
    "ground_truth": "정답",
    "gt_context": "참고 context"
  }
]
```

### 파싱된 응답 (`data/parsed_responses/parsed_*.json`)
```json
{
  "answer": "실제 답변",
  "context": "참고한 context"
}
```

## 평가 결과

평가 결과는 `results/` 폴더에 저장됩니다:
- `ragas_results_*.json`: 평가 결과 JSON 파일
- `ragas_results_*.jpg`: 평가 결과 시각화 이미지

## 라이선스

MIT License

