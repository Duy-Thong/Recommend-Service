import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set font for Vietnamese support
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_comparison_results(file_path):
    """Load comparison results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_metrics_comparison(results_data, output_dir='evaluation_data/visualizations'):
    """Create comparison plots for different metrics"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = results_data['results']
    methods = [r['method'] for r in results]
    threshold = results_data.get('threshold', 'N/A')

    # Define metrics to plot
    metrics = {
        'MRR': 'mrr',
        'NDCG@5': 'ndcg_at_5',
        'NDCG@10': 'ndcg_at_10',
        'NDCG@30': 'ndcg_at_30',
        'Hit Rate@5': 'hit_rate_at_5',
        'Hit Rate@10': 'hit_rate_at_10',
        'Hit Rate@30': 'hit_rate_at_30'
    }

    # 1. Bar chart for all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Recommendation System Evaluation (Threshold: {threshold})', fontsize=16, fontweight='bold')

    # Plot MRR
    ax = axes[0, 0]
    values = [r['mrr'] for r in results]
    bars = ax.bar(methods, values, color='steelblue', alpha=0.8)
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Mean Reciprocal Rank (MRR)', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # Plot NDCG metrics
    ax = axes[0, 1]
    x = np.arange(len(methods))
    width = 0.25
    ndcg5 = [r['ndcg_at_5'] for r in results]
    ndcg10 = [r['ndcg_at_10'] for r in results]
    ndcg30 = [r['ndcg_at_30'] for r in results]

    ax.bar(x - width, ndcg5, width, label='NDCG@5', color='coral', alpha=0.8)
    ax.bar(x, ndcg10, width, label='NDCG@10', color='lightseagreen', alpha=0.8)
    ax.bar(x + width, ndcg30, width, label='NDCG@30', color='orchid', alpha=0.8)

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('NDCG Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot Hit Rate metrics
    ax = axes[1, 0]
    hr5 = [r['hit_rate_at_5'] for r in results]
    hr10 = [r['hit_rate_at_10'] for r in results]
    hr30 = [r['hit_rate_at_30'] for r in results]

    ax.bar(x - width, hr5, width, label='Hit Rate@5', color='coral', alpha=0.8)
    ax.bar(x, hr10, width, label='Hit Rate@10', color='lightseagreen', alpha=0.8)
    ax.bar(x + width, hr30, width, label='Hit Rate@30', color='orchid', alpha=0.8)

    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Hit Rate Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Summary table
    ax = axes[1, 1]
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    table_data.append(['Method', 'MRR', 'NDCG@10', 'HR@10', 'Queries'])
    for r in results:
        table_data.append([
            r['method'][:15] + '...' if len(r['method']) > 15 else r['method'],
            f"{r['mrr']:.3f}",
            f"{r['ndcg_at_10']:.3f}",
            f"{r['hit_rate_at_10']:.3f}",
            str(r['num_queries'])
        ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'comparison_threshold_{threshold}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # 2. Radar chart for method comparison
    create_radar_chart(results_data, output_dir)

    # 3. Line plot showing metrics progression
    create_metrics_progression(results_data, output_dir)

def create_radar_chart(results_data, output_dir):
    """Create radar chart comparing all methods"""
    results = results_data['results']
    threshold = results_data.get('threshold', 'N/A')

    # Metrics for radar chart
    categories = ['MRR', 'NDCG@5', 'NDCG@10', 'Hit Rate@5', 'Hit Rate@10']
    metric_keys = ['mrr', 'ndcg_at_5', 'ndcg_at_10', 'hit_rate_at_5', 'hit_rate_at_10']

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']

    for idx, result in enumerate(results):
        values = [result[key] for key in metric_keys]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=result['method'],
                color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'Method Comparison - Radar Chart (Threshold: {threshold})',
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'radar_chart_threshold_{threshold}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_metrics_progression(results_data, output_dir):
    """Create line plot showing how metrics change at different k values"""
    results = results_data['results']
    threshold = results_data.get('threshold', 'N/A')

    k_values = [5, 10, 30]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Metrics Progression by K (Threshold: {threshold})',
                 fontsize=14, fontweight='bold')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']

    # NDCG progression
    for idx, result in enumerate(results):
        ndcg_values = [result['ndcg_at_5'], result['ndcg_at_10'], result['ndcg_at_30']]
        ax1.plot(k_values, ndcg_values, 'o-', linewidth=2, markersize=8,
                label=result['method'], color=colors[idx % len(colors)])

    ax1.set_xlabel('K (Top-K)', fontweight='bold')
    ax1.set_ylabel('NDCG Score', fontweight='bold')
    ax1.set_title('NDCG Progression', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    ax1.set_ylim([0, 1])

    # Hit Rate progression
    for idx, result in enumerate(results):
        hr_values = [result['hit_rate_at_5'], result['hit_rate_at_10'], result['hit_rate_at_30']]
        ax2.plot(k_values, hr_values, 'o-', linewidth=2, markersize=8,
                label=result['method'], color=colors[idx % len(colors)])

    ax2.set_xlabel('K (Top-K)', fontweight='bold')
    ax2.set_ylabel('Hit Rate', fontweight='bold')
    ax2.set_title('Hit Rate Progression', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    ax2.set_ylim([0, 1])

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'metrics_progression_threshold_{threshold}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def compare_thresholds(data_dir='evaluation_data'):
    """Compare results across different thresholds"""
    threshold_files = [
        'comparison_results_threshold_060.json',
        'comparison_results_threshold_065.json',
        'comparison_results_threshold_070.json',
        'comparison_results_threshold_075.json'
    ]

    threshold_data = {}
    for file in threshold_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            data = load_comparison_results(file_path)
            threshold = data.get('threshold', 'unknown')
            threshold_data[threshold] = data

    if not threshold_data:
        print("No threshold comparison files found")
        return

    # Plot comparison across thresholds for each method
    output_dir = os.path.join(data_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    # Get all methods from first threshold
    first_threshold = list(threshold_data.values())[0]
    methods = [r['method'] for r in first_threshold['results']]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Across Different Thresholds', fontsize=16, fontweight='bold')

    thresholds = sorted(threshold_data.keys())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']

    # MRR across thresholds
    ax = axes[0, 0]
    for idx, method in enumerate(methods):
        mrr_values = []
        for t in thresholds:
            result = next((r for r in threshold_data[t]['results'] if r['method'] == method), None)
            if result:
                mrr_values.append(result['mrr'])
            else:
                mrr_values.append(0)
        ax.plot(thresholds, mrr_values, 'o-', linewidth=2, markersize=8,
               label=method, color=colors[idx % len(colors)])
    ax.set_xlabel('Threshold', fontweight='bold')
    ax.set_ylabel('MRR', fontweight='bold')
    ax.set_title('MRR vs Threshold', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # NDCG@10 across thresholds
    ax = axes[0, 1]
    for idx, method in enumerate(methods):
        ndcg_values = []
        for t in thresholds:
            result = next((r for r in threshold_data[t]['results'] if r['method'] == method), None)
            if result:
                ndcg_values.append(result['ndcg_at_10'])
            else:
                ndcg_values.append(0)
        ax.plot(thresholds, ndcg_values, 'o-', linewidth=2, markersize=8,
               label=method, color=colors[idx % len(colors)])
    ax.set_xlabel('Threshold', fontweight='bold')
    ax.set_ylabel('NDCG@10', fontweight='bold')
    ax.set_title('NDCG@10 vs Threshold', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Hit Rate@10 across thresholds
    ax = axes[1, 0]
    for idx, method in enumerate(methods):
        hr_values = []
        for t in thresholds:
            result = next((r for r in threshold_data[t]['results'] if r['method'] == method), None)
            if result:
                hr_values.append(result['hit_rate_at_10'])
            else:
                hr_values.append(0)
        ax.plot(thresholds, hr_values, 'o-', linewidth=2, markersize=8,
               label=method, color=colors[idx % len(colors)])
    ax.set_xlabel('Threshold', fontweight='bold')
    ax.set_ylabel('Hit Rate@10', fontweight='bold')
    ax.set_title('Hit Rate@10 vs Threshold', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Number of queries across thresholds
    ax = axes[1, 1]
    for idx, method in enumerate(methods):
        num_queries = []
        for t in thresholds:
            result = next((r for r in threshold_data[t]['results'] if r['method'] == method), None)
            if result:
                num_queries.append(result['num_queries'])
            else:
                num_queries.append(0)
        ax.plot(thresholds, num_queries, 'o-', linewidth=2, markersize=8,
               label=method, color=colors[idx % len(colors)])
    ax.set_xlabel('Threshold', fontweight='bold')
    ax.set_ylabel('Number of Queries', fontweight='bold')
    ax.set_title('Query Count vs Threshold', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'threshold_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    """Main function to generate all visualizations"""
    data_dir = 'evaluation_data'

    print("Starting visualization generation...")
    print("=" * 60)

    # Visualize individual threshold results
    threshold_files = [
        'comparison_results_threshold_060.json',
        'comparison_results_threshold_065.json',
        'comparison_results_threshold_070.json',
        'comparison_results_threshold_075.json'
    ]

    for file in threshold_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"\nProcessing: {file}")
            data = load_comparison_results(file_path)
            plot_metrics_comparison(data)
        else:
            print(f"File not found: {file_path}")

    # Compare across thresholds
    print("\nGenerating threshold comparison...")
    compare_thresholds(data_dir)

    print("\n" + "=" * 60)
    print("Visualization generation completed!")
    print(f"Check the '{data_dir}/visualizations' directory for output files.")

if __name__ == '__main__':
    main()
