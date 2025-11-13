#!/usr/bin/env python3
"""
Dataset Performance Benchmark - So sánh performance của models trên các datasets khác nhau
Author: System Integration Team
Date: November 2024

Chạy: python -m scripts.dataset_benchmark
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Absolute imports
from utils.path_utils import get_project_root, get_datasets_dir, get_results_dir
from detection_system.models.ml_based.traditional_ml import TraditionalMLDetector
from detection_system.features.text_features.text_features import TextFeaturesExtractor

class DatasetBenchmark:
    def __init__(self):
        self.results = {}
        self.config = self._create_config()
        
    def _create_config(self):
        """Create config for benchmark"""
        class BenchmarkConfig:
            MAX_FEATURES = 1000
            MIN_DF = 1
            MAX_DF = 0.95
            NGRAM_RANGE = (1, 2)
            RANDOM_STATE = 42
            CV_FOLDS = 5
            PRIMARY_METRIC = 'f1_score'
            TEST_SIZE = 0.2
        return BenchmarkConfig
        
    def load_dataset(self, dataset_path, dataset_name):
        """Load and prepare dataset"""
        print(f"\n📊 Loading {dataset_name}...")
        
        df = pd.read_csv(dataset_path)
        
        # Clean data - remove rows with missing prompts
        original_size = len(df)
        df = df.dropna(subset=['prompt'])
        
        # Remove non-string prompts
        df = df[df['prompt'].apply(lambda x: isinstance(x, str))]
        
        cleaned_size = len(df)
        if cleaned_size != original_size:
            print(f"   ⚠️  Cleaned: {original_size} → {cleaned_size} samples (removed {original_size - cleaned_size} invalid)")
        
        # Sample large datasets for efficiency
        if len(df) > 10000:
            print(f"   ⚡ Sampling {len(df)} → 10000 for efficient processing...")
            df = df.sample(n=10000, random_state=42)
        
        # Convert labels to binary if needed
        if df['label'].dtype == 'object':
            labels = (df['label'] == 'malicious').astype(int)
        else:
            labels = df['label']
            
        # Basic statistics
        print(f"   Total samples: {len(df)}")
        print(f"   Malicious: {sum(labels)}")
        print(f"   Benign: {len(labels) - sum(labels)}")
        
        if 'difficulty' in df.columns:
            print(f"   Difficulty distribution:")
            for diff, count in df['difficulty'].value_counts().items():
                print(f"     {diff}: {count}")
        
        return df['prompt'].tolist(), labels.tolist(), df
    
    def evaluate_on_dataset(self, texts, labels, dataset_name, df_full=None):
        """Evaluate ML models on a specific dataset"""
        print(f"\n🔄 Evaluating on {dataset_name}...")
        
        # Initialize components
        feature_extractor = TextFeaturesExtractor(self.config)
        ml_detector = TraditionalMLDetector(self.config)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE, stratify=labels
        )
        
        print(f"   Training samples: {len(X_train_texts)}")
        print(f"   Test samples: {len(X_test_texts)}")
        
        # Extract features
        print("   Extracting features...")
        train_features = feature_extractor.extract_all_features(X_train_texts, fit=True)
        test_features = feature_extractor.extract_all_features(X_test_texts, fit=False)
        
        # Prepare feature matrices
        X_train = ml_detector.prepare_features(
            train_features['statistical_features'],
            train_features['tfidf_features']
        )
        
        X_test = ml_detector.prepare_features(
            test_features['statistical_features'], 
            test_features['tfidf_features']
        )
        
        print(f"   Feature matrix shape: {X_train.shape}")
        
        # Train models
        print("   Training models...")
        training_results = ml_detector.train_all_models(X_train, np.array(y_train))
        
        # Evaluate models
        print("   Evaluating models...")
        evaluation_results = ml_detector.evaluate_all_models(X_test, np.array(y_test))
        
        # Analyze by difficulty if available
        difficulty_analysis = None
        if df_full is not None and 'difficulty' in df_full.columns:
            difficulty_analysis = self.analyze_by_difficulty(
                X_test_texts, y_test, df_full, ml_detector, feature_extractor
            )
        
        # Store results
        self.results[dataset_name] = {
            'evaluation_results': evaluation_results,
            'training_results': training_results,
            'dataset_stats': {
                'total_samples': len(texts),
                'train_samples': len(X_train_texts),
                'test_samples': len(X_test_texts),
                'malicious_ratio': sum(labels) / len(labels),
                'feature_count': X_train.shape[1]
            },
            'difficulty_analysis': difficulty_analysis
        }
        
        # Print summary
        print(f"\n   📈 Results for {dataset_name}:")
        for model_name, metrics in evaluation_results.items():
            print(f"     {model_name}: F1={metrics['f1_score']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}")
        
        return evaluation_results
    
    def analyze_by_difficulty(self, test_texts, test_labels, df_full, ml_detector, feature_extractor):
        """Analyze performance by difficulty level"""
        print("   Analyzing by difficulty...")
        
        difficulty_results = {}
        
        # Group test samples by difficulty
        test_df = pd.DataFrame({
            'prompt': test_texts,
            'label': test_labels
        })
        
        # Match with original dataframe to get difficulty
        merged_df = test_df.merge(df_full[['prompt', 'difficulty']], on='prompt', how='left')
        
        for difficulty in merged_df['difficulty'].dropna().unique():
            if difficulty == 'realistic':  # Skip if no special difficulty
                continue
                
            diff_mask = merged_df['difficulty'] == difficulty
            diff_texts = merged_df[diff_mask]['prompt'].tolist()
            diff_labels = merged_df[diff_mask]['label'].tolist()
            
            if len(diff_texts) < 5:  # Skip if too few samples
                continue
                
            print(f"     Difficulty '{difficulty}': {len(diff_texts)} samples")
            
            # Extract features for this subset
            diff_features = feature_extractor.extract_all_features(diff_texts, fit=False)
            X_diff = ml_detector.prepare_features(
                diff_features['statistical_features'],
                diff_features['tfidf_features']
            )
            
            # Evaluate best model on this subset
            best_model_name = max(self.results.get('simple_dataset', {}).get('evaluation_results', {}), 
                                key=lambda x: self.results.get('simple_dataset', {}).get('evaluation_results', {}).get(x, {}).get('f1_score', 0), 
                                default='logistic_regression')
            
            if hasattr(ml_detector, 'trained_models') and best_model_name in ml_detector.trained_models:
                model = ml_detector.trained_models[best_model_name]['model']
                predictions = model.predict(X_diff)
                
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                difficulty_results[difficulty] = {
                    'sample_count': len(diff_texts),
                    'accuracy': accuracy_score(diff_labels, predictions),
                    'precision': precision_score(diff_labels, predictions, zero_division=0),
                    'recall': recall_score(diff_labels, predictions, zero_division=0),
                    'f1_score': f1_score(diff_labels, predictions, zero_division=0)
                }
                
                print(f"       F1: {difficulty_results[difficulty]['f1_score']:.4f}")
        
        return difficulty_results
    
    def run_comprehensive_benchmark(self):
        """Run benchmark on multiple datasets"""
        print("🚀 COMPREHENSIVE DATASET BENCHMARK")
        print("=" * 70)
        
        datasets_to_test = []
        datasets_dir = get_datasets_dir()
        
        # Look for challenging datasets
        challenging_files = list(datasets_dir.glob('challenging_dataset_*.csv'))
        if challenging_files:
            # Use the most recent challenging dataset
            latest_challenging = max(challenging_files, key=lambda x: x.stat().st_mtime)
            datasets_to_test.append(('Challenging Dataset', str(latest_challenging)))
        
        # Look for HuggingFace datasets
        huggingface_files = list(datasets_dir.glob('huggingface_dataset_*.csv'))
        if huggingface_files:
            # Use the most recent HuggingFace dataset
            latest_huggingface = max(huggingface_files, key=lambda x: x.stat().st_mtime)
            datasets_to_test.append(('HuggingFace Dataset', str(latest_huggingface)))
        
        if not datasets_to_test:
            print("❌ No datasets found in datasets/ directory")
            print("💡 Please ensure you have challenging_dataset_*.csv or huggingface_dataset_*.csv files")
            return {}
        
        print(f"📊 Found {len(datasets_to_test)} datasets to benchmark:")
        for name, path in datasets_to_test:
            print(f"   • {name}: {Path(path).name}")
        
        # Run evaluation on each dataset
        for dataset_name, dataset_path in datasets_to_test:
            try:
                texts, labels, df_full = self.load_dataset(dataset_path, dataset_name)
                self.evaluate_on_dataset(texts, labels, dataset_name.lower().replace(' ', '_'), df_full)
            except Exception as e:
                print(f"❌ Error evaluating {dataset_name}: {str(e)}")
                continue
        
        # Generate comparison report
        if self.results:
            self.generate_comparison_report()
        else:
            print("❌ No successful evaluations completed")
        
        return self.results
    
    def generate_comparison_report(self):
        """Generate detailed comparison report"""
        print(f"\n" + "=" * 60)
        print("📊 COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 60)
        
        if len(self.results) < 2:
            print("⚠️  Need at least 2 datasets to compare")
            return
        
        # Compare F1 scores across datasets
        print(f"\n🎯 F1 SCORE COMPARISON:")
        
        # Get all model names
        all_models = set()
        for dataset_results in self.results.values():
            all_models.update(dataset_results['evaluation_results'].keys())
        
        # Create comparison table
        comparison_data = []
        for model in sorted(all_models):
            row = {'Model': model.replace('_', ' ').title()}
            for dataset_name, results in self.results.items():
                f1_score = results['evaluation_results'].get(model, {}).get('f1_score', 0)
                row[dataset_name.replace('_', ' ').title()] = f1_score
            comparison_data.append(row)
        
        # Print table
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Performance drop analysis
        if 'simple_dataset' in self.results and 'challenging_dataset' in self.results:
            print(f"\n📉 PERFORMANCE DROP ANALYSIS:")
            simple_results = self.results['simple_dataset']['evaluation_results']
            challenging_results = self.results['challenging_dataset']['evaluation_results']
            
            for model in all_models:
                simple_f1 = simple_results.get(model, {}).get('f1_score', 0)
                challenging_f1 = challenging_results.get(model, {}).get('f1_score', 0)
                drop = simple_f1 - challenging_f1
                drop_percent = (drop / simple_f1 * 100) if simple_f1 > 0 else 0
                
                print(f"   {model.replace('_', ' ').title()}:")
                print(f"     Simple → Challenging: {simple_f1:.4f} → {challenging_f1:.4f}")
                print(f"     Drop: {drop:.4f} ({drop_percent:.1f}%)")
        
        # Difficulty analysis if available
        for dataset_name, results in self.results.items():
            if results['difficulty_analysis']:
                print(f"\n🎚️  DIFFICULTY ANALYSIS - {dataset_name.replace('_', ' ').title()}:")
                for difficulty, metrics in results['difficulty_analysis'].items():
                    print(f"   {difficulty.replace('_', ' ').title()}: "
                          f"F1={metrics['f1_score']:.4f} "
                          f"({metrics['sample_count']} samples)")
        
        # Dataset statistics comparison
        print(f"\n📋 DATASET STATISTICS:")
        for dataset_name, results in self.results.items():
            stats = results['dataset_stats']
            print(f"   {dataset_name.replace('_', ' ').title()}:")
            print(f"     Total samples: {stats['total_samples']}")
            print(f"     Features: {stats['feature_count']}")
            print(f"     Malicious ratio: {stats['malicious_ratio']:.2%}")
        
        # Save detailed report
        results_dir = get_results_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"dataset_benchmark_{timestamp}.json"
        report_path = results_dir / report_filename
        
        report_data = {
            'benchmark_date': datetime.now().isoformat(),
            'datasets_evaluated': list(self.results.keys()),
            'detailed_results': self.results,
            'summary': {
                'best_overall_model': self._find_best_model(),
                'most_challenging_difficulty': self._find_hardest_difficulty(),
                'key_insights': self._generate_insights()
            }
        }
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
            
            print(f"\n💾 Detailed benchmark report saved: {report_path}")
            
            # Also save to standard benchmark_report.json for compatibility
            standard_path = results_dir / "benchmark_report.json"
            with open(standard_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
            
        except Exception as e:
            print(f"\n❌ Failed to save benchmark report: {e}")
    
    def _find_best_model(self):
        """Find best performing model across datasets"""
        model_scores = {}
        
        for dataset_results in self.results.values():
            for model_name, metrics in dataset_results['evaluation_results'].items():
                if model_name not in model_scores:
                    model_scores[model_name] = []
                model_scores[model_name].append(metrics['f1_score'])
        
        # Calculate average F1 score
        avg_scores = {model: np.mean(scores) for model, scores in model_scores.items()}
        
        return max(avg_scores, key=avg_scores.get)
    
    def _find_hardest_difficulty(self):
        """Find most challenging difficulty level"""
        difficulty_scores = {}
        
        for dataset_results in self.results.values():
            if dataset_results['difficulty_analysis']:
                for difficulty, metrics in dataset_results['difficulty_analysis'].items():
                    if difficulty not in difficulty_scores:
                        difficulty_scores[difficulty] = []
                    difficulty_scores[difficulty].append(metrics['f1_score'])
        
        if not difficulty_scores:
            return None
        
        # Calculate average F1 score for each difficulty
        avg_scores = {diff: np.mean(scores) for diff, scores in difficulty_scores.items()}
        
        return min(avg_scores, key=avg_scores.get)
    
    def _generate_insights(self):
        """Generate key insights from benchmark"""
        insights = []
        
        # Check if there's significant performance drop
        if 'simple_dataset' in self.results and 'challenging_dataset' in self.results:
            simple_avg = np.mean([m['f1_score'] for m in self.results['simple_dataset']['evaluation_results'].values()])
            challenging_avg = np.mean([m['f1_score'] for m in self.results['challenging_dataset']['evaluation_results'].values()])
            
            drop = simple_avg - challenging_avg
            if drop > 0.1:
                insights.append(f"Significant performance drop on challenging dataset: {drop:.3f}")
            
            if challenging_avg < 0.8:
                insights.append("Models struggle with sophisticated attacks - need improvement")
            
            if simple_avg > 0.95:
                insights.append("Simple dataset may be too easy - perfect scores indicate overfitting")
        
        return insights

def main():
    """Run comprehensive benchmark"""
    benchmark = DatasetBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    print(f"\n✅ Benchmark completed!")
    print(f"📈 Check results/ folder for detailed analysis")
    
    return results

if __name__ == "__main__":
    main()
