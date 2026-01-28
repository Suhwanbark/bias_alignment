"""
Phase 4: DPO 훈련 전후 결과 비교

훈련 전/후의 sentiment와 bias 결과를 비교하여
DPO 훈련의 효과를 정량화

사용법:
    python compare_results.py \
        --before-sentiment data/sentiment_nvidia.json \
        --after-sentiment data/sentiment_nvidia_after.json \
        --before-bias result/before/nvidia_att_result.json \
        --after-bias result/after/nvidia_att_result.json \
        --output data/comparison_report.json
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime


def load_json(path: str) -> Optional[Dict]:
    """JSON 파일 로드"""
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_sentiment_metrics(data: Dict) -> Dict:
    """sentiment 결과에서 주요 메트릭 추출"""
    if not data:
        return {}

    results = data.get('results', [])
    correlation = data.get('correlation', {})

    # 평균 sentiment 계산
    sentiments = [r['avg_sentiment'] for r in results if r.get('avg_sentiment') is not None]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None

    return {
        "avg_sentiment": avg_sentiment,
        "pearson_r": correlation.get('pearson_r'),
        "p_value": correlation.get('p_value'),
        "n_samples": correlation.get('n_samples'),
        "interpretation": correlation.get('interpretation'),
    }


def extract_bias_metrics(data: Dict) -> Dict:
    """bias 결과에서 주요 메트릭 추출"""
    if not data:
        return {}

    # bias_attribute.py 결과 구조에 따라 파싱
    # 일반적으로 sector_bias, size_bias, composite_bias 등이 있음

    metrics = {}

    # 전체 bias index
    if 'bias_index' in data:
        metrics['bias_index'] = data['bias_index']
    elif 'composite_bias' in data:
        metrics['bias_index'] = data['composite_bias']

    # sector bias
    if 'sector_bias' in data:
        metrics['sector_bias'] = data['sector_bias']
    elif 'sector' in data:
        sector_data = data['sector']
        if isinstance(sector_data, dict) and 'composite' in sector_data:
            metrics['sector_bias'] = sector_data['composite']

    # size bias
    if 'size_bias' in data:
        metrics['size_bias'] = data['size_bias']
    elif 'size' in data:
        size_data = data['size']
        if isinstance(size_data, dict) and 'composite' in size_data:
            metrics['size_bias'] = size_data['composite']

    # buy rate 통계
    if 'ticker_stats' in data:
        ticker_stats = data['ticker_stats']
        buy_rates = [s.get('buy_rate', 0) for s in ticker_stats.values() if 'buy_rate' in s]
        if buy_rates:
            metrics['avg_buy_rate'] = sum(buy_rates) / len(buy_rates)
            metrics['min_buy_rate'] = min(buy_rates)
            metrics['max_buy_rate'] = max(buy_rates)

    return metrics


def calculate_change(before: Optional[float], after: Optional[float]) -> Dict:
    """변화량 계산"""
    if before is None or after is None:
        return {
            "before": before,
            "after": after,
            "change": None,
            "change_pct": None,
        }

    change = after - before
    change_pct = (change / abs(before) * 100) if before != 0 else None

    return {
        "before": before,
        "after": after,
        "change": change,
        "change_pct": f"{change_pct:+.1f}%" if change_pct is not None else None,
    }


def generate_comparison_report(
    before_sentiment: Dict,
    after_sentiment: Dict,
    before_bias: Dict,
    after_bias: Dict,
    target: str,
) -> Dict:
    """전후 비교 리포트 생성"""

    # 메트릭 추출
    before_sent = extract_sentiment_metrics(before_sentiment)
    after_sent = extract_sentiment_metrics(after_sentiment)
    before_b = extract_bias_metrics(before_bias)
    after_b = extract_bias_metrics(after_bias)

    # 비교 계산
    report = {
        "model": target,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "bias_index": calculate_change(
                before_b.get('bias_index'),
                after_b.get('bias_index')
            ),
            "sector_bias": calculate_change(
                before_b.get('sector_bias'),
                after_b.get('sector_bias')
            ),
            "size_bias": calculate_change(
                before_b.get('size_bias'),
                after_b.get('size_bias')
            ),
            "avg_sentiment": calculate_change(
                before_sent.get('avg_sentiment'),
                after_sent.get('avg_sentiment')
            ),
            "avg_buy_rate": calculate_change(
                before_b.get('avg_buy_rate'),
                after_b.get('avg_buy_rate')
            ),
            "correlation_sentiment_buyrate": calculate_change(
                before_sent.get('pearson_r'),
                after_sent.get('pearson_r')
            ),
        },
        "raw_data": {
            "before_sentiment": before_sent,
            "after_sentiment": after_sent,
            "before_bias": before_b,
            "after_bias": after_b,
        },
    }

    # 결론 생성
    conclusions = []

    # Bias 변화 평가
    bias_change = report['metrics']['bias_index']
    if bias_change['change'] is not None:
        if bias_change['change'] < -0.1:
            conclusions.append(f"✅ Bias index 크게 감소 ({bias_change['change_pct']})")
        elif bias_change['change'] < 0:
            conclusions.append(f"⚠️ Bias index 소폭 감소 ({bias_change['change_pct']})")
        else:
            conclusions.append(f"❌ Bias index 증가 또는 변화 없음 ({bias_change['change_pct']})")

    # Sentiment 변화 평가
    sent_change = report['metrics']['avg_sentiment']
    if sent_change['change'] is not None:
        if target == "nvidia":  # SELL bias → BUY로 교정: sentiment 증가 기대
            if sent_change['change'] > 0.1:
                conclusions.append(f"✅ Sentiment 긍정적으로 변화 ({sent_change['change']:+.3f})")
            elif sent_change['change'] > 0:
                conclusions.append(f"⚠️ Sentiment 소폭 개선 ({sent_change['change']:+.3f})")
            else:
                conclusions.append(f"❌ Sentiment 개선되지 않음 ({sent_change['change']:+.3f})")
        else:  # Qwen: BUY bias → SELL로 교정: sentiment 감소 기대
            if sent_change['change'] < -0.1:
                conclusions.append(f"✅ Sentiment 부정적으로 변화 ({sent_change['change']:+.3f})")
            elif sent_change['change'] < 0:
                conclusions.append(f"⚠️ Sentiment 소폭 변화 ({sent_change['change']:+.3f})")
            else:
                conclusions.append(f"❌ Sentiment 개선되지 않음 ({sent_change['change']:+.3f})")

    # Correlation 약화 평가
    corr_change = report['metrics']['correlation_sentiment_buyrate']
    if corr_change['before'] is not None and corr_change['after'] is not None:
        # 상관관계 절대값 감소 = 좋은 것 (bias와 sentiment 연결 약화)
        before_abs = abs(corr_change['before'])
        after_abs = abs(corr_change['after'])
        if after_abs < before_abs * 0.7:
            conclusions.append(f"✅ Sentiment-Bias 상관관계 크게 약화")
        elif after_abs < before_abs:
            conclusions.append(f"⚠️ Sentiment-Bias 상관관계 소폭 약화")
        else:
            conclusions.append(f"❌ Sentiment-Bias 상관관계 강화 또는 유지")

    report["conclusions"] = conclusions

    # 전체 평가
    success_count = sum(1 for c in conclusions if c.startswith("✅"))
    warning_count = sum(1 for c in conclusions if c.startswith("⚠️"))

    if success_count >= 2:
        report["overall"] = "DPO 훈련 효과 있음"
    elif success_count + warning_count >= 2:
        report["overall"] = "DPO 훈련 일부 효과 있음"
    else:
        report["overall"] = "DPO 훈련 효과 제한적"

    return report


def print_report(report: Dict):
    """리포트 출력"""
    print(f"\n{'='*60}")
    print(f"DPO 훈련 전후 비교 리포트")
    print(f"{'='*60}")
    print(f"모델: {report['model']}")
    print(f"시간: {report['timestamp']}")
    print()

    print(f"{'─'*60}")
    print(f"주요 메트릭 변화")
    print(f"{'─'*60}")

    metrics = report['metrics']
    for name, data in metrics.items():
        before = data.get('before')
        after = data.get('after')
        change_pct = data.get('change_pct', 'N/A')

        before_str = f"{before:.4f}" if before is not None else "N/A"
        after_str = f"{after:.4f}" if after is not None else "N/A"

        print(f"{name:30} {before_str:>10} → {after_str:>10}  ({change_pct})")

    print()
    print(f"{'─'*60}")
    print(f"결론")
    print(f"{'─'*60}")

    for conclusion in report.get('conclusions', []):
        print(f"  {conclusion}")

    print()
    print(f"전체 평가: {report.get('overall', 'N/A')}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="DPO 훈련 전후 결과 비교")
    parser.add_argument(
        "--before-sentiment",
        type=str,
        default=None,
        help="훈련 전 sentiment 분석 결과 JSON",
    )
    parser.add_argument(
        "--after-sentiment",
        type=str,
        default=None,
        help="훈련 후 sentiment 분석 결과 JSON",
    )
    parser.add_argument(
        "--before-bias",
        type=str,
        default=None,
        help="훈련 전 bias 분석 결과 JSON",
    )
    parser.add_argument(
        "--after-bias",
        type=str,
        default=None,
        help="훈련 후 bias 분석 결과 JSON",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nvidia",
        choices=["nvidia", "qwen"],
        help="대상 모델 (nvidia 또는 qwen)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/comparison_report.json",
        help="비교 리포트 출력 JSON 파일",
    )
    args = parser.parse_args()

    # 데이터 로드
    before_sentiment = load_json(args.before_sentiment) if args.before_sentiment else {}
    after_sentiment = load_json(args.after_sentiment) if args.after_sentiment else {}
    before_bias = load_json(args.before_bias) if args.before_bias else {}
    after_bias = load_json(args.after_bias) if args.after_bias else {}

    # 로드된 데이터 확인
    loaded_count = sum([
        bool(before_sentiment),
        bool(after_sentiment),
        bool(before_bias),
        bool(after_bias),
    ])

    if loaded_count == 0:
        print("경고: 로드된 데이터가 없습니다.")
        print("최소 하나 이상의 전/후 데이터 파일을 지정하세요.")
        return

    print(f"\n로드된 데이터:")
    print(f"  before-sentiment: {'✓' if before_sentiment else '✗'} {args.before_sentiment or '(미지정)'}")
    print(f"  after-sentiment:  {'✓' if after_sentiment else '✗'} {args.after_sentiment or '(미지정)'}")
    print(f"  before-bias:      {'✓' if before_bias else '✗'} {args.before_bias or '(미지정)'}")
    print(f"  after-bias:       {'✓' if after_bias else '✗'} {args.after_bias or '(미지정)'}")

    # 리포트 생성
    report = generate_comparison_report(
        before_sentiment=before_sentiment,
        after_sentiment=after_sentiment,
        before_bias=before_bias,
        after_bias=after_bias,
        target=args.target,
    )

    # 출력
    print_report(report)

    # 저장
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"리포트 저장: {args.output}")


if __name__ == "__main__":
    main()
