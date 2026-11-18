"""
Master Experiment Runner.
Executes all 5 NLP experiments sequentially or selectively.

Usage:
    python run_all_experiments.py              # Run all experiments (1 through 5)
    python run_all_experiments.py --exp 1      # Run only Experiment 1 (EDA)
    python run_all_experiments.py --exp 2 4 5  # Run selected experiments
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

EXPERIMENT_MAP = {
    1: ('01_eda.py', 'Exploratory Data Analysis & Lexical Profiling'),
    2: ('02_preprocessing_comparison.py', 'Standard vs. Indian-Context Preprocessing Benchmarking'),
    3: ('03_feature_comparison.py', 'TF-IDF vs. Dense Embeddings (Empirical Justification)'),
    4: ('04_model_comparison.py', 'Base (NB) vs. Proposed (LR & SVM) Cross-Validation & ROC Evaluation'),
    5: ('05_lime_explainability.py', 'LIME Local Attribution & Qualitative Error Analysis')
}


def run_single_experiment(exp_num: int) -> bool:
    if exp_num not in EXPERIMENT_MAP:
        print(f"[-] Invalid experiment number: {exp_num}. Choose from 1 to 5.")
        return False

    script_name, description = EXPERIMENT_MAP[exp_num]
    script_path = PROJECT_ROOT / 'experiments' / script_name

    print("\n" + "="*80)
    print(f"  EXECUTION: Experiment {exp_num} - {description}")
    print(f"  Script: {script_path.relative_to(PROJECT_ROOT)}")
    print("="*80 + "\n")

    start_time = time.time()
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT))
    duration = time.time() - start_time

    if result.returncode == 0:
        print(f"\n[+] Experiment {exp_num} finished successfully in {duration:.2f} seconds.")
        return True
    else:
        print(f"\n[-] Experiment {exp_num} failed with exit code {result.returncode}.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run NLP Fake News Detection Experiments")
    parser.add_argument('--exp', nargs='+', type=int, default=[1, 2, 3, 4, 5],
                        help="List of experiment numbers to execute (1-5)")
    args = parser.parse_args()

    (PROJECT_ROOT / 'results').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'data').mkdir(parents=True, exist_ok=True)

    print("\n" + "#"*80)
    print("  EXPLAINABLE FAKE NEWS DETECTION IN THE INDIAN CONTEXT")
    print("  Final-Year NLP Experimental Testbed")
    print("#"*80)

    overall_start = time.time()
    summary = {}

    for exp_id in sorted(args.exp):
        success = run_single_experiment(exp_id)
        summary[exp_id] = success

    total_duration = time.time() - overall_start

    print("\n" + "="*80)
    print("  EXPERIMENTAL SUITE SUMMARY")
    print("="*80)
    all_passed = True
    for exp_id, status in summary.items():
        script_name, desc = EXPERIMENT_MAP[exp_id]
        status_str = "PASSED" if status else "FAILED"
        marker = "[+]" if status else "[-]"
        print(f"  {marker} Exp {exp_id} ({desc}): {status_str}")
        if not status:
            all_passed = False

    print(f"\n  Total Execution Time: {total_duration:.2f} seconds")
    print(f"  All outputs, graphics, and JSON metrics saved to: {PROJECT_ROOT / 'results'}")
    print("="*80 + "\n")

    if not all_passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
