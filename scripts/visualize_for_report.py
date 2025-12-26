"""
Generate visualizations for the evaluation report.

Creates figures that match the report structure:
1. Baseline Comparison (Table 5.2)
2. Ablation Study (Table 5.3)
3. Ranking Metrics Results (Table 5.1)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configure matplotlib for better quality
plt.rcParams['font.size'] = 11
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def load_results(file_path):
    """Load evaluation results from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_baseline_comparison(data, output_dir):
    """
    Create visualization for baseline comparison.

    Corresponds to Table 5.2 in report:
    - Ngẫu nhiên (Random)
    - TF-IDF + Cosine
    - SimCSE (chỉ tiêu đề) → cascade_1layer
    - Lọc phân tầng (đề xuất) → cascade_3layer
    """
    results = data['results']

    # Map methods to report names
    method_mapping = {
        'Ngẫu nhiên': 'Ngẫu nhiên',
        'TF-IDF + Cosine': 'TF-IDF + Cosin',
        'jaccard': 'Jaccard',
        'Cascade (3 vòng lọc)': 'Lọc phân tầng (đề xuất)'
    }

    # Filter and reorder results
    baseline_results = []
    for key, name in method_mapping.items():
        result = next((r for r in results if r['method'] == key), None)
        if result:
            baseline_results.append({
                'method': name,
                'mrr': result['mrr'],
                'ndcg_at_10': result['ndcg_at_10'],
                'hit_rate_at_10': result['hit_rate_at_10']
            })

    if not baseline_results:
        print("Warning: No baseline results found")
        return

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('So sánh với phương pháp cơ sở', fontsize=14, fontweight='bold', y=1.02)

    methods = [r['method'] for r in baseline_results]
    colors = ['#FF6B6B', '#FFA07A', '#45B7D1', '#4ECDC4']

    # MRR comparison
    ax = axes[0]
    mrr_values = [r['mrr'] for r in baseline_results]
    bars = ax.bar(range(len(methods)), mrr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('MRR', fontweight='bold')
    ax.set_title('Mean Reciprocal Rank', fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # NDCG@10 comparison
    ax = axes[1]
    ndcg_values = [r['ndcg_at_10'] for r in baseline_results]
    bars = ax.bar(range(len(methods)), ndcg_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('NDCG@10', fontweight='bold')
    ax.set_title('Normalized DCG at 10', fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Hit Rate@10 comparison
    ax = axes[2]
    hr_values = [r['hit_rate_at_10'] for r in baseline_results]
    bars = ax.bar(range(len(methods)), hr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Hit Rate@10', fontweight='bold')
    ax.set_title('Hit Rate at 10', fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'baseline_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_ablation_study(data, output_dir):
    """
    Create visualization for ablation study.

    Corresponds to Table 5.3 in report:
    - Vòng 1 (tiêu đề) → cascade_1layer
    - Vòng 1 + 2 (+ kinh nghiệm) → cascade_2layer (if exists)
    - Vòng 1 + 2 + 3 (+ kỹ năng) → cascade_3layer
    """
    results = data['results']

    # Map cascade layers to report names
    layer_mapping = {
        'cascade_1layer': 'Vòng 1\n(tiêu đề)',
        'cascade_2layer': 'Vòng 1 + 2\n(+ kinh nghiệm)',
        'Cascade (3 vòng lọc)': 'Vòng 1 + 2 + 3\n(+ kỹ năng)'
    }

    ablation_results = []
    for key, name in layer_mapping.items():
        result = next((r for r in results if r['method'] == key), None)
        if result:
            ablation_results.append({
                'config': name,
                'ndcg_at_10': result['ndcg_at_10'],
                'hit_rate_at_10': result['hit_rate_at_10']
            })

    if not ablation_results:
        print("Warning: No ablation study results found")
        return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Ảnh hưởng của các vòng lọc', fontsize=14, fontweight='bold', y=1.02)

    configs = [r['config'] for r in ablation_results]
    colors = ['#FFB6C1', '#87CEEB', '#98FB98']

    # NDCG@10 progression
    ax = axes[0]
    ndcg_values = [r['ndcg_at_10'] for r in ablation_results]
    bars = ax.bar(range(len(configs)), ndcg_values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.2)
    ax.set_ylabel('NDCG@10', fontweight='bold')
    ax.set_title('NDCG@10 theo cấu hình', fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Hit Rate@10 progression
    ax = axes[1]
    hr_values = [r['hit_rate_at_10'] for r in ablation_results]
    bars = ax.bar(range(len(configs)), hr_values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Hit Rate@10', fontweight='bold')
    ax.set_title('Hit Rate@10 theo cấu hình', fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ablation_study.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_ranking_metrics_table(data, output_dir):
    """
    Create visualization for main ranking metrics results.

    Corresponds to Table 5.1 in report - shows the best method's performance.
    Uses cascade_3layer as the proposed method.
    """
    results = data['results']

    # Get the best method (Cascade 3 vòng lọc)
    best_result = next((r for r in results if r['method'] == 'Cascade (3 vòng lọc)'), None)

    if not best_result:
        print("Warning: No Cascade (3 vòng lọc) results found")
        return

    # Create bar chart for all metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Kết quả độ đo xếp hạng - Phương pháp đề xuất',
                 fontsize=14, fontweight='bold')

    metrics = {
        'MRR': best_result['mrr'],
        'NDCG@5': best_result['ndcg_at_5'],
        'NDCG@10': best_result['ndcg_at_10'],
        'Hit Rate@5': best_result['hit_rate_at_5'],
        'Hit Rate@10': best_result['hit_rate_at_10']
    }

    metric_names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77', '#4D96FF']

    bars = ax.bar(metric_names, values, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.2)
    ax.set_ylabel('Giá trị', fontweight='bold')
    ax.set_title('Các độ đo xếp hạng', fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ranking_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_combined_overview(data, output_dir):
    """Create a comprehensive overview comparing all methods."""
    results = data['results']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    methods = [r['method'] for r in results]
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    # 1. MRR comparison
    ax1 = fig.add_subplot(gs[0, 0])
    mrr_values = [r['mrr'] for r in results]
    bars = ax1.barh(methods, mrr_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('MRR', fontweight='bold')
    ax1.set_title('Mean Reciprocal Rank', fontweight='bold')
    ax1.set_xlim([0, 1.0])
    ax1.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # 2. NDCG progression (k=5, 10, 30)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(methods))
    width = 0.25
    ax2.bar(x - width, [r['ndcg_at_5'] for r in results], width,
            label='NDCG@5', color='coral', alpha=0.8)
    ax2.bar(x, [r['ndcg_at_10'] for r in results], width,
            label='NDCG@10', color='lightseagreen', alpha=0.8)
    ax2.bar(x + width, [r['ndcg_at_30'] for r in results], width,
            label='NDCG@30', color='orchid', alpha=0.8)
    ax2.set_ylabel('NDCG', fontweight='bold')
    ax2.set_title('NDCG tại các giá trị K', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. Hit Rate progression (k=5, 10, 30)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - width, [r['hit_rate_at_5'] for r in results], width,
            label='HR@5', color='coral', alpha=0.8)
    ax3.bar(x, [r['hit_rate_at_10'] for r in results], width,
            label='HR@10', color='lightseagreen', alpha=0.8)
    ax3.bar(x + width, [r['hit_rate_at_30'] for r in results], width,
            label='HR@30', color='orchid', alpha=0.8)
    ax3.set_ylabel('Hit Rate', fontweight='bold')
    ax3.set_title('Hit Rate tại các giá trị K', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
    ax3.set_ylim([0, 1.0])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. Summary radar chart
    ax4 = fig.add_subplot(gs[1, 1], projection='polar')
    categories = ['MRR', 'NDCG@10', 'HR@10']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for idx, result in enumerate(results):
        values = [result['mrr'], result['ndcg_at_10'], result['hit_rate_at_10']]
        values += values[:1]
        ax4.plot(angles, values, 'o-', linewidth=2, label=result['method'],
                color=colors[idx])
        ax4.fill(angles, values, alpha=0.15, color=colors[idx])

    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, size=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('So sánh tổng quan', fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax4.grid(True)

    plt.suptitle('Tổng quan đánh giá hệ thống gợi ý', fontsize=16, fontweight='bold', y=0.98)

    output_path = os.path.join(output_dir, 'combined_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations for the report."""
    print("=" * 70)
    print("GENERATING VISUALIZATIONS FOR REPORT")
    print("=" * 70)

    # Use the best threshold (0.75) for main visualizations
    data_file = 'evaluation_data/comparison_results_threshold_075.json'
    output_dir = 'evaluation_data/visualizations/report'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return

    print(f"\nLoading data from: {data_file}")
    data = load_results(data_file)
    print(f"Found {len(data['results'])} methods")

    # Generate visualizations
    print("\nGenerating visualizations...")
    print("-" * 70)

    print("\n1. Baseline Comparison (for Table 5.2)")
    create_baseline_comparison(data, output_dir)

    print("\n2. Ablation Study (for Table 5.3)")
    create_ablation_study(data, output_dir)

    print("\n3. Ranking Metrics (for Table 5.1)")
    create_ranking_metrics_table(data, output_dir)

    print("\n4. Combined Overview")
    create_combined_overview(data, output_dir)

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS GENERATED!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
