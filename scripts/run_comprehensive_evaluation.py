"""
Run comprehensive evaluation with multiple thresholds and configurations.
This will create more varied results for better visualization.
"""

import sys
import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_evaluation(threshold: float, output_suffix: str = ""):
    """Run evaluation with specific threshold."""

    # Define methods and their mapping
    methods_config = {
        'basic': {
            'methods': ['random', 'tfidf', 'jaccard', 'word2vec'],
            'output': f'comparison_results_threshold_{str(threshold).replace(".", "")}{output_suffix}.json'
        },
        'cascade': {
            'methods': ['cascade_1layer', 'cascade_2layer', 'cascade_3layer'],
            'output': f'cascade_results_threshold_{str(threshold).replace(".", "")}{output_suffix}.json'
        },
        'all': {
            'methods': ['random', 'tfidf', 'jaccard', 'word2vec', 'cascade_1layer', 'cascade_2layer', 'cascade_3layer'],
            'output': f'comparison_results_all_threshold_{str(threshold).replace(".", "")}{output_suffix}.json'
        }
    }

    for config_name, config in methods_config.items():
        output_file = f"./evaluation_data/{config['output']}"

        logger.info(f"\n{'='*70}")
        logger.info(f"Running {config_name} evaluation with threshold={threshold}")
        logger.info(f"Methods: {config['methods']}")
        logger.info(f"Output: {output_file}")
        logger.info(f"{'='*70}\n")

        cmd = [
            sys.executable,
            "scripts/evaluate_from_csv.py",
            "--threshold", str(threshold),
            "--methods", *config['methods'],
            "--output", output_file
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            logger.info(f"✓ Successfully completed {config_name} evaluation")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to run {config_name} evaluation: {e}")
            continue


def main():
    """Run evaluations with multiple thresholds."""

    # Different thresholds to test
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    logger.info("="*70)
    logger.info("COMPREHENSIVE EVALUATION SUITE")
    logger.info("="*70)
    logger.info(f"Thresholds to evaluate: {thresholds}")
    logger.info("="*70)

    for threshold in thresholds:
        logger.info(f"\n\n{'#'*70}")
        logger.info(f"# THRESHOLD: {threshold}")
        logger.info(f"{'#'*70}\n")

        run_evaluation(threshold)

    logger.info("\n\n" + "="*70)
    logger.info("ALL EVALUATIONS COMPLETED!")
    logger.info("="*70)
    logger.info("\nGenerated files:")

    eval_dir = Path("./evaluation_data")
    json_files = sorted(eval_dir.glob("*.json"))
    for f in json_files:
        if f.name.startswith("comparison_") or f.name.startswith("cascade_"):
            logger.info(f"  - {f.name}")

    logger.info("\nNext steps:")
    logger.info("  1. Run: python visualize_evaluation.py")
    logger.info("  2. Check: evaluation_data/visualizations/")
    logger.info("="*70)


if __name__ == "__main__":
    main()
