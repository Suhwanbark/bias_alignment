"""
Step 2: 이벤트 조합으로 DPO 데이터셋 생성

사용법:
    python generate_dpo_dataset.py --events data/events_nvidia.json --num-samples 1000 --output data/dpo_nvidia.jsonl
    python generate_dpo_dataset.py --events data/events_qwen.json --num-samples 1000 --output data/dpo_qwen.jsonl
"""

import os
import json
import argparse
import random
from typing import Dict, List, Tuple
from tqdm import tqdm
import concurrent.futures

from llm_client import VLLMClient
from config import (
    DEFAULT_MODEL_ID,
    DEFAULT_VLLM_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DPO_SAMPLES_PER_TICKER_NVIDIA,
    DPO_SAMPLES_PER_TICKER_QWEN,
    DPO_GENERATION_SYSTEM_PROMPT,
    DPO_GENERATION_USER_PROMPT,
)


def load_events(events_path: str) -> Dict:
    """이벤트 JSON 파일 로드"""
    with open(events_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sample_balanced_evidence(
    positive_events: List[str],
    negative_events: List[str],
    num_positive: int = 2,
    num_negative: int = 2,
) -> Tuple[List[str], List[str]]:
    """균형잡힌 evidence 샘플링 (긍정 + 부정)"""
    pos_sample = random.sample(positive_events, min(num_positive, len(positive_events)))
    neg_sample = random.sample(negative_events, min(num_negative, len(negative_events)))
    return pos_sample, neg_sample


def format_evidence(positive: List[str], negative: List[str]) -> str:
    """evidence를 프롬프트 형식으로 포맷"""
    all_evidence = []
    for ev in positive:
        all_evidence.append(f"[+] {ev}")
    for ev in negative:
        all_evidence.append(f"[-] {ev}")
    random.shuffle(all_evidence)
    return "\n".join(all_evidence)


def generate_dpo_sample(
    client: VLLMClient,
    ticker: str,
    company_name: str,
    positive_events: List[str],
    negative_events: List[str],
    bias_direction: str,  # "sell" for NVIDIA (bias toward sell), "buy" for Qwen (bias toward buy)
) -> Dict:
    """
    단일 DPO 샘플 생성

    Args:
        client: LLM 클라이언트
        ticker: 주식 ticker
        company_name: 회사명
        positive_events: 긍정 이벤트 목록
        negative_events: 부정 이벤트 목록
        bias_direction: 현재 모델의 bias 방향 ("sell" 또는 "buy")

    Returns:
        DPO 형식의 dict (prompt, chosen, rejected)
    """
    # 균형잡힌 evidence 샘플링
    pos_sample, neg_sample = sample_balanced_evidence(positive_events, negative_events)
    evidence_str = format_evidence(pos_sample, neg_sample)

    # LLM으로 buy/sell 응답 생성
    prompt = DPO_GENERATION_USER_PROMPT.format(
        ticker=ticker,
        company_name=company_name,
        evidence=evidence_str,
    )

    result = client.get_json_response(
        prompt=prompt,
        system_prompt=DPO_GENERATION_SYSTEM_PROMPT,
    )

    if not result or 'buy_response' not in result or 'sell_response' not in result:
        return None

    # DPO 형식 생성
    # bias 방향과 반대되는 것을 chosen으로 설정
    # NVIDIA (sell bias): buy를 chosen으로
    # Qwen (buy bias): sell을 chosen으로
    dpo_prompt = f"{ticker}. Evidence: {evidence_str}. Should you buy or sell?"

    if bias_direction == "sell":
        # SELL bias 모델 -> BUY를 chosen으로 (bias 교정)
        chosen = result['buy_response']
        rejected = result['sell_response']
    else:
        # BUY bias 모델 -> SELL을 chosen으로 (bias 교정)
        chosen = result['sell_response']
        rejected = result['buy_response']

    return {
        "prompt": dpo_prompt,
        "chosen": chosen,
        "rejected": rejected,
        "ticker": ticker,
        "company_name": company_name,
        "evidence_positive": pos_sample,
        "evidence_negative": neg_sample,
    }


def main():
    parser = argparse.ArgumentParser(description="DPO 데이터셋 생성")
    parser.add_argument(
        "--events",
        type=str,
        required=True,
        help="이벤트 JSON 파일 경로 (generate_events.py 출력)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"생성용 모델 ID (기본값: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=DEFAULT_VLLM_URL,
        help=f"vLLM 서버 URL (기본값: {DEFAULT_VLLM_URL})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="총 생성할 DPO 샘플 수 (기본값: 1000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 JSONL 파일 경로",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="최대 동시 워커 수 (기본값: 20)",
    )
    args = parser.parse_args()

    # 이벤트 데이터 로드
    print(f"{args.events}에서 이벤트 데이터 로드 중...")
    events_data = load_events(args.events)

    target_model = events_data['target_model']
    tickers = events_data['tickers']

    # 유효한 ticker만 필터링 (이벤트가 있는 것)
    valid_tickers = [
        t for t in tickers
        if 'error' not in t
        and len(t.get('positive_events', [])) >= 2
        and len(t.get('negative_events', [])) >= 2
    ]

    if not valid_tickers:
        print("오류: 유효한 ticker가 없습니다.")
        return

    print(f"유효한 ticker: {len(valid_tickers)}개")

    # bias 방향 결정
    if target_model == "nvidia":
        bias_direction = "sell"  # NVIDIA는 SELL bias
        samples_per_ticker = DPO_SAMPLES_PER_TICKER_NVIDIA
        default_output = "data/dpo_nvidia.jsonl"
    else:
        bias_direction = "buy"  # Qwen은 BUY bias
        samples_per_ticker = DPO_SAMPLES_PER_TICKER_QWEN
        default_output = "data/dpo_qwen.jsonl"

    output_path = args.output or default_output

    # ticker당 샘플 수 계산
    total_samples = args.num_samples
    samples_per_ticker = total_samples // len(valid_tickers)
    remaining = total_samples % len(valid_tickers)

    print(f"대상 모델: {target_model}")
    print(f"Bias 방향: {bias_direction}")
    print(f"총 샘플 수: {total_samples}")
    print(f"Ticker당 샘플: ~{samples_per_ticker}개")

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 클라이언트 초기화
    client = VLLMClient(
        model_id=args.model_id,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=DEFAULT_TOP_P,
        base_url=args.vllm_url,
    )

    print(f"모델: {args.model_id}")
    print(f"vLLM 서버: {args.vllm_url}")

    # 각 ticker에 대해 샘플 수 할당
    ticker_samples = []
    for i, ticker_data in enumerate(valid_tickers):
        count = samples_per_ticker + (1 if i < remaining else 0)
        for _ in range(count):
            ticker_samples.append(ticker_data)

    random.shuffle(ticker_samples)
    print(f"\n총 {len(ticker_samples)}개 샘플 생성 시작...")

    # DPO 샘플 생성
    all_samples = []
    failed_count = 0

    def process_sample(ticker_data):
        return generate_dpo_sample(
            client=client,
            ticker=ticker_data['ticker'],
            company_name=ticker_data['company_name'],
            positive_events=ticker_data['positive_events'],
            negative_events=ticker_data['negative_events'],
            bias_direction=bias_direction,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_sample, td) for td in ticker_samples]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="DPO 샘플 생성",
        ):
            try:
                result = future.result()
                if result:
                    all_samples.append(result)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"예외: {e}")
                failed_count += 1

    # JSONL로 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # 요약
    print(f"\n{'─'*50}")
    print(f"DPO 데이터셋 저장 완료: {output_path}")
    print(f"성공: {len(all_samples)}개")
    print(f"실패: {failed_count}개")
    print(f"Bias 교정 방향: {bias_direction} -> {'buy' if bias_direction == 'sell' else 'sell'}")
    print(f"{'─'*50}")

    # 통계
    ticker_counts = {}
    for sample in all_samples:
        ticker = sample['ticker']
        ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

    print(f"\nTicker별 샘플 수:")
    for ticker, count in sorted(ticker_counts.items()):
        print(f"  {ticker}: {count}개")


if __name__ == "__main__":
    main()
