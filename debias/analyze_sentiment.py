"""
Phase 1: Description의 sentiment 분석 및 buy_rate와의 correlation 계산

recall_descriptions.py로 수집한 description들의 sentiment를 분석하고
buy_rate와의 상관관계를 계산하여 가설 검증

사용법:
    python analyze_sentiment.py \
        --input data/recall_nvidia.json \
        --output data/sentiment_nvidia.json \
        --plot data/correlation_nvidia.png
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from scipy import stats

# Sentiment 분석 라이브러리
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("경고: vaderSentiment 미설치. pip install vaderSentiment")

# 시각화 라이브러리
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("경고: matplotlib 미설치. 시각화 비활성화")


def analyze_sentiment_vader(text: str, analyzer: 'SentimentIntensityAnalyzer') -> float:
    """VADER를 사용하여 sentiment 점수 계산 (-1 ~ +1)"""
    scores = analyzer.polarity_scores(text)
    return scores['compound']


def analyze_sentiment_simple(text: str) -> float:
    """간단한 키워드 기반 sentiment 분석 (VADER 없을 경우)"""
    positive_words = [
        'strong', 'growth', 'opportunity', 'excellent', 'leading', 'innovative',
        'profitable', 'positive', 'upside', 'attractive', 'solid', 'robust',
        'outperform', 'buy', 'bullish', 'promising', 'advantage', 'momentum'
    ]
    negative_words = [
        'weak', 'decline', 'risk', 'concern', 'challenging', 'struggling',
        'loss', 'negative', 'downside', 'expensive', 'uncertain', 'volatile',
        'underperform', 'sell', 'bearish', 'threat', 'headwind', 'pressure'
    ]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def analyze_all_descriptions(
    results: List[Dict],
    analyzer: Optional['SentimentIntensityAnalyzer'] = None,
) -> List[Dict]:
    """모든 ticker의 description에 대해 sentiment 분석"""
    analyzed_results = []

    for ticker_data in results:
        ticker = ticker_data['ticker']
        descriptions = ticker_data.get('descriptions', [])

        sentiments = []
        for desc in descriptions:
            response = desc.get('response')
            if response:
                if analyzer:
                    score = analyze_sentiment_vader(response, analyzer)
                else:
                    score = analyze_sentiment_simple(response)
                sentiments.append(score)

        if sentiments:
            avg_sentiment = np.mean(sentiments)
            std_sentiment = np.std(sentiments)
        else:
            avg_sentiment = None
            std_sentiment = None

        result = {
            "ticker": ticker,
            "company_name": ticker_data.get('company_name'),
            "sector": ticker_data.get('sector'),
            "buy_rate": ticker_data.get('buy_rate'),
            "sentiments": sentiments,
            "avg_sentiment": avg_sentiment,
            "std_sentiment": std_sentiment,
            "num_descriptions": len(sentiments),
        }

        analyzed_results.append(result)

    return analyzed_results


def calculate_correlation(
    analyzed_results: List[Dict],
) -> Dict:
    """buy_rate와 sentiment 간의 상관관계 계산"""
    # buy_rate가 있는 ticker만 필터링
    valid_data = [
        r for r in analyzed_results
        if r.get('buy_rate') is not None and r.get('avg_sentiment') is not None
    ]

    if len(valid_data) < 3:
        return {
            "pearson_r": None,
            "p_value": None,
            "n_samples": len(valid_data),
            "interpretation": "데이터 부족 (최소 3개 필요)",
        }

    buy_rates = [r['buy_rate'] for r in valid_data]
    sentiments = [r['avg_sentiment'] for r in valid_data]

    # Pearson 상관계수
    r, p = stats.pearsonr(buy_rates, sentiments)

    # 해석
    if p >= 0.05:
        interpretation = "통계적으로 유의하지 않음 (p >= 0.05)"
    elif abs(r) >= 0.7:
        direction = "양의" if r > 0 else "음의"
        interpretation = f"강한 {direction} 상관관계"
    elif abs(r) >= 0.4:
        direction = "양의" if r > 0 else "음의"
        interpretation = f"중간 정도의 {direction} 상관관계"
    else:
        interpretation = "약한 상관관계"

    return {
        "pearson_r": float(r),
        "p_value": float(p),
        "n_samples": len(valid_data),
        "interpretation": interpretation,
    }


def create_correlation_plot(
    analyzed_results: List[Dict],
    output_path: str,
    model_id: str,
    correlation: Dict,
) -> bool:
    """buy_rate vs sentiment scatter plot 생성"""
    if not MATPLOTLIB_AVAILABLE:
        return False

    valid_data = [
        r for r in analyzed_results
        if r.get('buy_rate') is not None and r.get('avg_sentiment') is not None
    ]

    if len(valid_data) < 2:
        return False

    buy_rates = [r['buy_rate'] for r in valid_data]
    sentiments = [r['avg_sentiment'] for r in valid_data]
    tickers = [r['ticker'] for r in valid_data]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    scatter = ax.scatter(buy_rates, sentiments, alpha=0.7, s=100, c='steelblue', edgecolors='white')

    # 각 점에 ticker 라벨 추가
    for i, ticker in enumerate(tickers):
        ax.annotate(
            ticker,
            (buy_rates[i], sentiments[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8,
        )

    # 회귀선
    if len(valid_data) >= 2:
        z = np.polyfit(buy_rates, sentiments, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(buy_rates), max(buy_rates), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'회귀선 (r={correlation["pearson_r"]:.3f})')

    ax.set_xlabel('Buy Rate', fontsize=12)
    ax.set_ylabel('Average Sentiment Score', fontsize=12)
    ax.set_title(f'Buy Rate vs Description Sentiment\n{model_id}', fontsize=14)

    # 상관계수 텍스트
    r = correlation.get('pearson_r')
    p_val = correlation.get('p_value')
    if r is not None:
        text = f"Pearson r = {r:.3f}\np-value = {p_val:.4f}\nn = {correlation['n_samples']}"
        ax.text(
            0.05, 0.95, text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return True


def main():
    parser = argparse.ArgumentParser(description="Description sentiment 분석 및 correlation 계산")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="recall_descriptions.py 출력 JSON 파일",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="sentiment 분석 결과 JSON 파일",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="correlation scatter plot 이미지 파일",
    )
    parser.add_argument(
        "--use-simple",
        action="store_true",
        help="VADER 대신 간단한 키워드 기반 분석 사용",
    )
    args = parser.parse_args()

    # 입력 파일 로드
    print(f"\n{'─'*60}")
    print(f"Sentiment 분석")
    print(f"{'─'*60}")

    with open(args.input, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    model_id = input_data.get('model_id', 'unknown')
    target = input_data.get('target', 'unknown')
    results = input_data.get('results', [])

    print(f"입력 파일: {args.input}")
    print(f"모델: {model_id}")
    print(f"대상: {target}")
    print(f"ticker 수: {len(results)}개")

    # 출력 경로 설정
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = base.replace('recall_', 'sentiment_') + '.json'
        if args.output == args.input:
            args.output = base + '_sentiment.json'

    if args.plot is None:
        base = os.path.splitext(args.output)[0]
        args.plot = base.replace('sentiment_', 'correlation_') + '.png'

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    # Sentiment 분석기 초기화
    analyzer = None
    if VADER_AVAILABLE and not args.use_simple:
        analyzer = SentimentIntensityAnalyzer()
        print(f"분석 방법: VADER")
    else:
        print(f"분석 방법: Simple keyword-based")

    # Sentiment 분석
    print(f"\nSentiment 분석 중...")
    analyzed_results = analyze_all_descriptions(results, analyzer)

    # 상관관계 계산
    correlation = calculate_correlation(analyzed_results)

    print(f"\n{'─'*40}")
    print(f"상관관계 분석 결과")
    print(f"{'─'*40}")
    print(f"샘플 수: {correlation['n_samples']}")
    if correlation['pearson_r'] is not None:
        print(f"Pearson r: {correlation['pearson_r']:.4f}")
        print(f"p-value: {correlation['p_value']:.4f}")
    print(f"해석: {correlation['interpretation']}")
    print(f"{'─'*40}")

    # 결과 저장
    output_data = {
        "model_id": model_id,
        "target": target,
        "timestamp": datetime.now().isoformat(),
        "analysis_method": "vader" if analyzer else "simple_keywords",
        "results": analyzed_results,
        "correlation": correlation,
        "summary": {
            "total_tickers": len(analyzed_results),
            "tickers_with_sentiment": sum(1 for r in analyzed_results if r['avg_sentiment'] is not None),
            "tickers_with_buy_rate": sum(1 for r in analyzed_results if r['buy_rate'] is not None),
            "avg_sentiment_all": float(np.mean([r['avg_sentiment'] for r in analyzed_results if r['avg_sentiment'] is not None])) if any(r['avg_sentiment'] is not None for r in analyzed_results) else None,
        },
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nsentiment 결과 저장: {args.output}")

    # 시각화
    if MATPLOTLIB_AVAILABLE:
        plot_created = create_correlation_plot(
            analyzed_results,
            args.plot,
            model_id,
            correlation,
        )
        if plot_created:
            print(f"상관관계 플롯 저장: {args.plot}")
        else:
            print(f"플롯 생성 실패 (데이터 부족)")
    else:
        print(f"matplotlib 미설치로 시각화 건너뜀")

    # 가설 검증 결과 출력
    print(f"\n{'─'*60}")
    print(f"가설 검증 결과")
    print(f"{'─'*60}")

    if correlation['pearson_r'] is not None:
        r = correlation['pearson_r']
        p = correlation['p_value']

        if target == "nvidia":
            # NVIDIA: SELL bias → 부정 sentiment 예상
            if r > 0.4 and p < 0.05:
                print("✅ 가설 지지: buy_rate와 sentiment 간 양의 상관관계")
                print("   → buy_rate가 낮은(SELL bias) ticker는 부정적 sentiment")
            elif r < -0.4 and p < 0.05:
                print("❌ 가설 기각: 예상과 반대 방향의 상관관계")
            else:
                print("⚠️  약한 상관관계 또는 유의하지 않음")
        else:
            # Qwen: BUY bias → 긍정 sentiment 예상
            if r > 0.4 and p < 0.05:
                print("✅ 가설 지지: buy_rate와 sentiment 간 양의 상관관계")
                print("   → buy_rate가 높은(BUY bias) ticker는 긍정적 sentiment")
            elif r < -0.4 and p < 0.05:
                print("❌ 가설 기각: 예상과 반대 방향의 상관관계")
            else:
                print("⚠️  약한 상관관계 또는 유의하지 않음")
    else:
        print("⚠️  상관관계 계산 불가 (데이터 부족)")

    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
