#!/usr/bin/env python3
"""
HuggingFace Dataset Performance Test - Test model performance trên large-scale dataset
Author: System Integration Team
Date: November 2024

Chạy: python -m scripts.huggingface_test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import os

# Absolute imports
from utils.path_utils import get_project_root, get_datasets_dir, get_results_dir
from detection_system.models.ml_based.traditional_ml import TraditionalMLDetector
from detection_system.features.text_features.text_features import TextFeaturesExtractor

class HuggingFaceModelTester:
    def __init__(self):
        self.config = self._create_config()
        self.results = {}

    def _create_config(self):
        class HFConfig:
            MAX_FEATURES = 5000  # Increase for large dataset
            MIN_DF = 5  # Higher threshold for large dataset
            MAX_DF = 0.95
            NGRAM_RANGE = (1, 2)
            RANDOM_STATE = 42
            CV_FOLDS = 3
            PRIMARY_METRIC = 'f1_score'
            TEST_SIZE = 0.1  # Smaller test size for efficiency
        return HFConfig

    def find_huggingface_dataset(self):
        """Tìm dataset HuggingFace mới nhất"""
        datasets_dir = get_datasets_dir()

        # Find HuggingFace dataset files
        hf_files = list(datasets_dir.glob('huggingface_dataset_*.csv'))

        if not hf_files:
            print("No HuggingFace dataset found. Run huggingface_downloader.py first.")
            return None

        # Use the most recent file
        latest_file = max(hf_files, key=os.path.getctime)
        return str(latest_file)

    def load_and_sample_dataset(self, dataset_path, sample_size=10000):
        """Load và sample dataset để test nhanh"""
        print(f"Loading HuggingFace dataset: {dataset_path}")

        df = pd.read_csv(dataset_path)

        print(f"Original dataset size: {len(df)}")
        print(f"Label distribution:")
        print(df['label'].value_counts())

        # Sample for faster testing if dataset is very large
        if len(df) > sample_size:
            print(f"\nSampling {sample_size} samples for testing...")
            # Stratified sampling to maintain label balance
            df_sample = df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size//2), random_state=42)
            ).reset_index(drop=True)
        else:
            df_sample = df

        print(f"Testing on {len(df_sample)} samples")
        print(f"Sampled label distribution:")
        print(df_sample['label'].value_counts())

        return df_sample

    def test_on_huggingface_dataset(self, df, dataset_name="HuggingFace"):
        """Test models trên HuggingFace dataset"""
        print(f"\nTesting on {dataset_name} Dataset")
        print("-" * 60)

        # Prepare data
        texts = df['prompt'].tolist()
        labels = df['label'].tolist()

        # Convert string labels if needed
        if isinstance(labels[0], str):
            labels = [(1 if label == 'malicious' else 0) for label in labels]

        print(f"Samples: {len(texts)}")
        print(f"Malicious: {sum(labels)} ({sum(labels)/len(labels):.1%})")

        # Initialize components
        feature_extractor = TextFeaturesExtractor(self.config)
        ml_detector = TraditionalMLDetector(self.config)

        # Split data
        from sklearn.model_selection import train_test_split

        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=labels
        )

        print(f"Train: {len(X_train_texts)}, Test: {len(X_test_texts)}")

        # Extract features
        print("Extracting features...")
        train_features = feature_extractor.extract_all_features(X_train_texts, fit=True)
        test_features = feature_extractor.extract_all_features(X_test_texts, fit=False)

        X_train = ml_detector.prepare_features(
            train_features['statistical_features'],
            train_features['tfidf_features']
        )
        X_test = ml_detector.prepare_features(
            test_features['statistical_features'],
            test_features['tfidf_features']
        )

        print(f"Feature matrix: {X_train.shape}")

        # Train models
        print("Training models...")
        training_results = ml_detector.train_all_models(X_train, np.array(y_train))

        # Evaluate models
        print("Evaluating models...")
        evaluation_results = ml_detector.evaluate_all_models(X_test, np.array(y_test))

        # Print results
        print(f"\nResults:")
        for model_name, metrics in evaluation_results.items():
            print(f" {model_name.replace('_', ' ').title()}:")
            print(f"   F1: {metrics['f1_score']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")
            print(f"   AUC: {metrics['auc_score']:.4f}")

        # Store results
        self.results[dataset_name] = {
            'evaluation_results': evaluation_results,
            'dataset_stats': {
                'total_samples': len(texts),
                'train_samples': len(X_train_texts),
                'test_samples': len(X_test_texts),
                'malicious_ratio': sum(labels) / len(labels),
                'feature_count': X_train.shape[1]
            }
        }

        return evaluation_results

    def analyze_dataset_characteristics(self, df):
        """Phân tích đặc điểm của HuggingFace dataset"""
        print(f"\nDATASET ANALYSIS")
        print("-" * 40)

        # Text length analysis
        text_lengths = df['prompt'].str.len()
        word_counts = df['prompt'].str.split().str.len()

        print(f"Text Length Statistics:")
        print(f"  Mean: {text_lengths.mean():.1f}")
        print(f"  Median: {text_lengths.median():.1f}")
        print(f"  Min: {text_lengths.min()}")
        print(f"  Max: {text_lengths.max()}")

        print(f"Word Count Statistics:")
        print(f"  Mean: {word_counts.mean():.1f}")
        print(f"  Median: {word_counts.median():.1f}")

        # Attack type analysis
        if 'attack_type' in df.columns:
            print(f"\nAttack Type Distribution:")
            attack_dist = df['attack_type'].value_counts()
            for attack_type, count in attack_dist.head(10).items():
                percentage = count / len(df) * 100
                print(f"  {attack_type}: {count} ({percentage:.1f}%)")

        # Source analysis if available
        if 'source' in df.columns:
            print(f"\nSource Distribution:")
            source_dist = df['source'].value_counts()
            for source, count in source_dist.head(5).items():
                percentage = count / len(df) * 100
                print(f"  {source[:50]}...: {count} ({percentage:.1f}%)")

        # Label by attack type
        if 'attack_type' in df.columns:
            print(f"\nMalicious samples by attack type:")
            malicious_df = df[df['label'] == 'malicious']
            if len(malicious_df) > 0:
                mal_attack_dist = malicious_df['attack_type'].value_counts()
                for attack_type, count in mal_attack_dist.head(5).items():
                    percentage = count / len(malicious_df) * 100
                    print(f"  {attack_type}: {count} ({percentage:.1f}%)")

    def compare_with_other_datasets(self):
        """So sánh với datasets khác đã test trước đó"""
        print(f"\nPERFORMANCE COMPARISON")
        print("=" * 60)

        # Load previous results if available
        results_file = get_results_dir() / 'benchmark_report.json'
        previous_results = {}

        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    report = json.load(f)
                previous_results = report.get('detailed_results', {})
            except:
                pass

        # Combine with current results
        all_results = {**previous_results, **self.results}

        # Create comparison table
        if len(all_results) > 1:
            print(f"\nF1 Score Comparison:")
            print("-" * 80)

            models = ['logistic_regression', 'random_forest', 'svm', 'gradient_boosting']
            datasets = list(all_results.keys())

            # Header
            header = f"{'Model':<20}"
            for dataset in datasets:
                header += f" {dataset[:12]:<14}"
            print(header)
            print("-" * 80)

            # Data rows
            for model in models:
                row = f"{model.replace('_', ' ').title():<20}"
                for dataset in datasets:
                    if dataset in all_results and 'evaluation_results' in all_results[dataset]:
                        eval_results = all_results[dataset]['evaluation_results']
                        if model in eval_results:
                            f1_score = eval_results[model]['f1_score']
                            row += f" {f1_score:.4f}      "
                        else:
                            row += f" {'N/A':<14}"
                    else:
                        row += f" {'N/A':<14}"
                print(row)

        # Dataset size comparison
        print(f"\nDataset Characteristics:")
        print("-" * 60)
        for dataset_name, results in all_results.items():
            if 'dataset_stats' in results:
                stats = results['dataset_stats']
                print(f"{dataset_name}:")
                print(f"  Samples: {stats['total_samples']:,}")
                print(f"  Malicious: {stats['malicious_ratio']:.1%}")
                print(f"  Features: {stats['feature_count']:,}")

    def run_huggingface_test(self, sample_size=10000):
        """Chạy test trên HuggingFace dataset"""
        print("HUGGINGFACE DATASET PERFORMANCE TEST")
        print("=" * 70)

        # Find dataset
        dataset_path = self.find_huggingface_dataset()
        if not dataset_path:
            return False

        # Load and sample data
        df = self.load_and_sample_dataset(dataset_path, sample_size)

        # Analyze characteristics
        self.analyze_dataset_characteristics(df)

        # Test models
        evaluation_results = self.test_on_huggingface_dataset(df, "HuggingFace_Large")

        # Compare with other datasets
        self.compare_with_other_datasets()

        # Save results
        self.save_results()

        print(f"\nHuggingFace dataset testing completed!")
        return True

    def save_results(self):
        """Lưu kết quả test"""
        results_dir = get_results_dir()
        results_dir.mkdir(exist_ok=True)

        report = {
            'test_date': datetime.now().isoformat(),
            'dataset_source': 'HuggingFace: ahsanayub/malicious-prompts',
            'results': self.results
        }

        report_path = results_dir / 'huggingface_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Results saved to: {report_path}")

def main():
    """Main function"""
    tester = HuggingFaceModelTester()

    # Test with different sample sizes
    print("Choose testing mode:")
    print("1. Quick test (1,000 samples)")
    print("2. Medium test (10,000 samples)")
    print("3. Full test (50,000+ samples)")

    try:
        choice = input("Enter choice (1-3): ").strip()

        if choice == '1':
            sample_size = 1000
        elif choice == '2':
            sample_size = 10000
        elif choice == '3':
            sample_size = 50000
        else:
            print("Using default: Medium test (10,000 samples)")
            sample_size = 10000

    except:
        print("Using default: Medium test (10,000 samples)")
        sample_size = 10000

    success = tester.run_huggingface_test(sample_size)

    if success:
        print(f"\nKEY INSIGHTS:")
        print(" • HuggingFace dataset provides large-scale realistic testing")
        print(" • Compare performance with smaller synthetic datasets")
        print(" • Check for overfitting vs real-world generalization")
        print(" • Use this for final model validation")

    return success

if __name__ == "__main__":
    main()
