#!/usr/bin/env python3
"""
Comprehensive Model Testing Suite - Test models trên multiple datasets
Author: System Integration Team
Date: November 2024

Chạy: python -m scripts.comprehensive_test_suite
"""

import pandas as pd
import numpy as np
import        print("COMPREHENSIVE TESTING REPORT")
        print("="*60)
        
        if not all_results:
            print("No test results available")
            returnfrom pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Absolute imports
from utils.path_utils import get_project_root, get_datasets_dir, get_results_dir
from detection_system.models.ml_based.traditional_ml import TraditionalMLDetector
from detection_system.features.text_features.text_features import TextFeaturesExtractor

class ComprehensiveModelTester:
    def __init__(self):
        self.results = {}
        self.config = self._create_config()
        
    def _create_config(self):
        class TestConfig:
            MAX_FEATURES = 1000
            MIN_DF = 1
            MAX_DF = 0.95
            NGRAM_RANGE = (1, 2)
            RANDOM_STATE = 42
            CV_FOLDS = 3
            PRIMARY_METRIC = 'f1_score'
            TEST_SIZE = 0.2
        return TestConfig
        
    def find_datasets(self):
        """Tìm tất cả datasets có sẵn"""
        datasets_dir = Path('../datasets')
        
        datasets = {
            'simple': None,
            'challenging': None,
            'realworld': None
        }
        
        # Simple dataset
        simple_path = datasets_dir / 'full_dataset.csv'
        if simple_path.exists():
            datasets['simple'] = str(simple_path)
            
        # Challenging dataset (newest)
        challenging_files = list(datasets_dir.glob('challenging_dataset_*.csv'))
        if challenging_files:
            datasets['challenging'] = str(max(challenging_files, key=os.path.getctime))
            
        # Real-world dataset (newest) 
        realworld_files = list(datasets_dir.glob('realworld_dataset_*.csv'))
        if realworld_files:
            datasets['realworld'] = str(max(realworld_files, key=os.path.getctime))
            
        return datasets
        
    def test_on_dataset(self, dataset_path, dataset_name):
        """Test model trên một dataset"""
        print(f"\nTesting on {dataset_name} Dataset")
        print("-" * 50)
        
        # Load data
        df = pd.read_csv(dataset_path)
        texts = df['prompt'].tolist()
        
        if df['label'].dtype == 'object':
            labels = (df['label'] == 'malicious').astype(int).tolist()
        else:
            labels = df['label'].tolist()
            
        print(f"Loaded {len(texts)} samples ({sum(labels)} malicious, {len(labels) - sum(labels)} benign)")
        
        # Initialize components
        feature_extractor = TextFeaturesExtractor(self.config)
        ml_detector = TraditionalMLDetector(self.config)
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE, stratify=labels
        )
        
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
        
        print(f"Feature shape: {X_train.shape}")
        
        # Train and evaluate
        print("Training models...")
        training_results = ml_detector.train_all_models(X_train, np.array(y_train))
        
        print("Evaluating models...")
        evaluation_results = ml_detector.evaluate_all_models(X_test, np.array(y_test))
        
        # Detailed analysis
        detailed_analysis = self.analyze_predictions(
            X_test_texts, y_test, df, ml_detector, evaluation_results
        )
        
        # Store results
        self.results[dataset_name] = {
            'evaluation_results': evaluation_results,
            'dataset_stats': {
                'total_samples': len(texts),
                'malicious_count': sum(labels),
                'benign_count': len(labels) - sum(labels),
                'feature_count': X_train.shape[1],
                'train_samples': len(X_train_texts),
                'test_samples': len(X_test_texts)
            },
            'detailed_analysis': detailed_analysis
        }
        
        # Print summary
        print(f"\nResults Summary for {dataset_name}:")
        for model_name, metrics in evaluation_results.items():
            print(f"  {model_name.replace('_', ' ').title()}:")
            print(f"    F1: {metrics['f1_score']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")  
            print(f"    Recall: {metrics['recall']:.4f}")
            
        return evaluation_results
        
    def analyze_predictions(self, test_texts, test_labels, original_df, ml_detector, evaluation_results):
        """Phân tích chi tiết predictions"""
        analysis = {}
        
        # Get best model
        best_model_name = max(evaluation_results.keys(), 
                            key=lambda x: evaluation_results[x]['f1_score'])
        best_model = ml_detector.trained_models[best_model_name]['model']
        
        # Extract features for test set
        feature_extractor = TextFeaturesExtractor(self.config)
        test_features = feature_extractor.extract_all_features(test_texts, fit=False)
        X_test = ml_detector.prepare_features(
            test_features['statistical_features'],
            test_features['tfidf_features']
        )
        
        # Get predictions and probabilities
        predictions = best_model.predict(X_test)
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_test)[:, 1]
        else:
            probabilities = best_model.decision_function(X_test)
            # Normalize to [0, 1]
            probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
        
        # Analyze errors
        errors = []
        correct = []
        
        for i, (text, true_label, pred_label, prob) in enumerate(zip(test_texts, test_labels, predictions, probabilities)):
            result = {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'true_label': true_label,
                'predicted_label': pred_label,
                'probability': prob,
                'correct': true_label == pred_label
            }
            
            if true_label == pred_label:
                correct.append(result)
            else:
                errors.append(result)
        
        # Categorize errors
        false_positives = [e for e in errors if e['true_label'] == 0 and e['predicted_label'] == 1]
        false_negatives = [e for e in errors if e['true_label'] == 1 and e['predicted_label'] == 0]
        
        analysis = {
            'best_model': best_model_name,
            'total_predictions': len(predictions),
            'correct_predictions': len(correct),
            'errors': len(errors),
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'error_examples': {
                'false_positives': false_positives[:5],  # Top 5 examples
                'false_negatives': false_negatives[:5]
            },
            'probability_distribution': {
                'mean': float(np.mean(probabilities)),
                'std': float(np.std(probabilities)),
                'min': float(np.min(probabilities)),
                'max': float(np.max(probabilities))
            }
        }
        
        return analysis
        
    def run_comprehensive_test(self):
        """Chạy test trên tất cả datasets"""
        print("COMPREHENSIVE MODEL TESTING SUITE")
        print("=" * 60)
        
        # Find available datasets
        datasets = self.find_datasets()
        
        print("Available datasets:")
        for name, path in datasets.items():
            if path:
                print(f"  {name.title()}: {path}")
            else:
                print(f"  {name.title()}: Not found")
        
        # Test on each available dataset
        for name, path in datasets.items():
            if path:
                try:
                    self.test_on_dataset(path, name)
                except Exception as e:
                    print(f"Error testing on {name}: {str(e)}")
                    continue
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        # Create visualizations
        self.create_visualizations()
        
        return self.results
        
    def generate_comprehensive_report(self):
        """Tạo báo cáo tổng hợp"""
        print(f"\n" + "=" * 60)
        print("📈 COMPREHENSIVE TESTING REPORT")
        print("=" * 60)
        
        if not self.results:
            print("❌ No test results available")
            return
            
        # Performance comparison table
        print(f"\nPERFORMANCE COMPARISON ACROSS DATASETS:")
        
        # Create comparison DataFrame
        comparison_data = []
        datasets = list(self.results.keys())
        models = set()
        
        for dataset_results in self.results.values():
            models.update(dataset_results['evaluation_results'].keys())
        
        for model in sorted(models):
            row = {'Model': model.replace('_', ' ').title()}
            for dataset in datasets:
                if model in self.results[dataset]['evaluation_results']:
                    f1_score = self.results[dataset]['evaluation_results'][model]['f1_score']
                    row[dataset.title()] = f1_score
                else:
                    row[dataset.title()] = 0.0
            comparison_data.append(row)
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Dataset difficulty analysis
        print(f"\nDATASET DIFFICULTY ANALYSIS:")
        for dataset_name, results in self.results.items():
            stats = results['dataset_stats']
            avg_f1 = np.mean([m['f1_score'] for m in results['evaluation_results'].values()])
            
            print(f"\n  {dataset_name.title()} Dataset:")
            print(f"    Average F1 Score: {avg_f1:.4f}")
            print(f"    Samples: {stats['total_samples']}")
            print(f"    Features: {stats['feature_count']}")
            print(f"    Malicious Ratio: {stats['malicious_count']/stats['total_samples']:.2%}")
            
            if avg_f1 > 0.95:
                print(f"    💚 Assessment: Easy (possible overfitting)")
            elif avg_f1 > 0.85:
                print(f"    💛 Assessment: Moderate difficulty")
            elif avg_f1 > 0.75:
                print(f"    🧡 Assessment: Challenging")
            else:
                print(f"    ❤️ Assessment: Very difficult")
        
        # Error analysis
        print(f"\nERROR ANALYSIS:")
        for dataset_name, results in self.results.items():
            if 'detailed_analysis' in results:
                analysis = results['detailed_analysis']
                print(f"\n  {dataset_name.title()} Dataset ({analysis['best_model']}):")
                print(f"    False Positives: {analysis['false_positives']}")
                print(f"    False Negatives: {analysis['false_negatives']}")
                print(f"    Accuracy: {analysis['correct_predictions']}/{analysis['total_predictions']} ({analysis['correct_predictions']/analysis['total_predictions']:.2%})")
                
                # Show error examples
                if analysis['error_examples']['false_positives']:
                    print(f"    Sample False Positives:")
                    for i, fp in enumerate(analysis['error_examples']['false_positives'][:3]):
                        print(f"      {i+1}. \"{fp['text']}\" (confidence: {fp['probability']:.3f})")
                        
                if analysis['error_examples']['false_negatives']:
                    print(f"    Sample False Negatives:")
                    for i, fn in enumerate(analysis['error_examples']['false_negatives'][:3]):
                        print(f"      {i+1}. \"{fn['text']}\" (confidence: {fn['probability']:.3f})")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        
        # Check for overfitting on simple dataset
        if 'simple' in self.results:
            simple_avg = np.mean([m['f1_score'] for m in self.results['simple']['evaluation_results'].values()])
            if simple_avg > 0.98:
                print("  Models may be overfitting on simple synthetic data")
        
        # Compare performance drops
        if 'simple' in self.results and 'challenging' in self.results:
            simple_f1 = np.mean([m['f1_score'] for m in self.results['simple']['evaluation_results'].values()])
            challenging_f1 = np.mean([m['f1_score'] for m in self.results['challenging']['evaluation_results'].values()])
            drop = simple_f1 - challenging_f1
            
            if drop > 0.15:
                print(f"  📉 Significant performance drop on challenging dataset ({drop:.3f})")
                print("      → Consider more robust feature engineering")
                print("      → Train on more diverse data")
        
        if 'realworld' in self.results:
            realworld_f1 = np.mean([m['f1_score'] for m in self.results['realworld']['evaluation_results'].values()])
            if realworld_f1 < 0.8:
                print(f"  Real-world performance needs improvement ({realworld_f1:.3f})")
                print("      → Focus on reducing false positives")
                print("      → Improve context understanding")
        
        # Save detailed report
        self.save_detailed_report()
        
    def create_visualizations(self):
        """Tạo visualizations"""
        if not self.results or len(self.results) < 2:
            return
            
        print(f"\nCreating visualizations...")
        
        # Create results directory
        results_dir = Path('../results')
        results_dir.mkdir(exist_ok=True)
        
        # 1. Performance comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # F1 scores comparison
        datasets = list(self.results.keys())
        models = ['logistic_regression', 'random_forest', 'svm', 'gradient_boosting']
        
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, model in enumerate(models):
            f1_scores = []
            for dataset in datasets:
                if model in self.results[dataset]['evaluation_results']:
                    f1_scores.append(self.results[dataset]['evaluation_results'][model]['f1_score'])
                else:
                    f1_scores.append(0)
            
            ax1.bar(x + i * width, f1_scores, width, label=model.replace('_', ' ').title())
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels([d.title() for d in datasets])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Dataset difficulty visualization
        difficulties = []
        sample_counts = []
        labels = []
        
        for dataset_name, results in self.results.items():
            avg_f1 = np.mean([m['f1_score'] for m in results['evaluation_results'].values()])
            difficulties.append(1 - avg_f1)  # Higher value = more difficult
            sample_counts.append(results['dataset_stats']['total_samples'])
            labels.append(dataset_name.title())
        
        scatter = ax2.scatter(sample_counts, difficulties, s=200, alpha=0.7, c=range(len(datasets)), cmap='viridis')
        
        for i, label in enumerate(labels):
            ax2.annotate(label, (sample_counts[i], difficulties[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Difficulty (1 - Average F1)')
        ax2.set_title('Dataset Size vs Difficulty')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'comprehensive_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Performance charts saved to {results_dir / 'comprehensive_performance_analysis.png'}")
        
    def save_detailed_report(self):
        """Lưu báo cáo chi tiết"""
        results_dir = Path('../results')
        results_dir.mkdir(exist_ok=True)
        
        report = {
            'test_date': datetime.now().isoformat(),
            'datasets_tested': list(self.results.keys()),
            'summary': {
                'total_datasets': len(self.results),
                'best_performing_dataset': None,
                'most_challenging_dataset': None,
                'recommended_improvements': []
            },
            'detailed_results': self.results
        }
        
        # Find best/worst datasets
        if self.results:
            dataset_scores = {}
            for dataset_name, results in self.results.items():
                avg_f1 = np.mean([m['f1_score'] for m in results['evaluation_results'].values()])
                dataset_scores[dataset_name] = avg_f1
            
            report['summary']['best_performing_dataset'] = max(dataset_scores, key=dataset_scores.get)
            report['summary']['most_challenging_dataset'] = min(dataset_scores, key=dataset_scores.get)
        
        report_path = results_dir / 'comprehensive_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Detailed report saved to {report_path}")

def main():
    """Main function"""
    tester = ComprehensiveModelTester()
    results = tester.run_comprehensive_test()
    
    print(f"\nComprehensive testing completed!")
    print(f"Check the results/ folder for detailed analysis and visualizations")
    
    return results

if __name__ == "__main__":
    main()
