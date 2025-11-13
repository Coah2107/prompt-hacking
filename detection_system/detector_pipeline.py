"""
Main Detection Pipeline - Integrate tất cả components thành một system hoàn chỉnh
Author: System Integration Team
Date: November 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

# Absolute imports
from detection_system.config import Config
from detection_system.features.text_features.text_features import TextFeaturesExtractor
from detection_system.models.rule_based.pattern_detector import RuleBasedDetector
from detection_system.models.ml_based.traditional_ml import TraditionalMLDetector

class DetectionPipeline:
    def __init__(self, config=None):
        self.config = config or Config
        self.feature_extractor = TextFeaturesExtractor(self.config)
        self.rule_detector = RuleBasedDetector()
        self.ml_detector = TraditionalMLDetector(self.config)
        
    def load_data(self):
        """
        Load training và test data
        Lý do: Centralized data loading với error handling
        """
        try:
            print("Loading datasets...")
            
            train_df = pd.read_csv(self.config.TRAIN_DATA)
            test_df = pd.read_csv(self.config.TEST_DATA)
            
            print(f"Train set: {len(train_df)} samples")
            print(f"Test set: {len(test_df)} samples")
            
            # Convert labels to binary
            train_labels = (train_df['label'] == 'malicious').astype(int)
            test_labels = (test_df['label'] == 'malicious').astype(int)
            
            return {
                'train_texts': train_df['prompt'].tolist(),
                'train_labels': train_labels.tolist(),
                'test_texts': test_df['prompt'].tolist(), 
                'test_labels': test_labels.tolist(),
                'train_df': train_df,
                'test_df': test_df
            }
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def extract_features(self, data):
        """
        Extract features cho training và testing
        """
        print("\n=== FEATURE EXTRACTION ===")
        
        # Extract training features
        print("Extracting training features...")
        train_features = self.feature_extractor.extract_all_features(
            data['train_texts'], fit=True
        )
        
        # Extract test features
        print("Extracting test features...")
        test_features = self.feature_extractor.extract_all_features(
            data['test_texts'], fit=False
        )
        
        return train_features, test_features
    
    def evaluate_rule_based(self, data):
        """
        Evaluate rule-based detector
        """
        print("\n=== RULE-BASED DETECTION ===")
        
        # Test on test set
        start_time = time.time()
        results = self.rule_detector.detect_batch(data['test_texts'])
        end_time = time.time()
        
        # Evaluate performance
        evaluation = self.rule_detector.evaluate_predictions(
            results, [data['test_df'].iloc[i]['label'] for i in range(len(results))]
        )
        
        evaluation['inference_time'] = end_time - start_time
        evaluation['avg_time_per_sample'] = evaluation['inference_time'] / len(data['test_texts'])
        
        print(f"Rule-based Results:")
        print(f"  Precision: {evaluation['precision']:.4f}")
        print(f"  Recall: {evaluation['recall']:.4f}")
        print(f"  F1 Score: {evaluation['f1_score']:.4f}")
        print(f"  Inference time: {evaluation['inference_time']:.2f}s")
        
        return evaluation, results
    
    def train_and_evaluate_ml(self, data, train_features, test_features):
        """
        Train và evaluate ML models
        """
        print("\n=== ML-BASED DETECTION ===")
        
        # Prepare features
        X_train = self.ml_detector.prepare_features(
            train_features['statistical_features'],
            train_features['tfidf_features']
        )
        
        X_test = self.ml_detector.prepare_features(
            test_features['statistical_features'],
            test_features['tfidf_features']
        )
        
        y_train = np.array(data['train_labels'])
        y_test = np.array(data['test_labels'])
        
        # Train all models
        print(f"Training features shape: {X_train.shape}")
        training_results = self.ml_detector.train_all_models(X_train, y_train)
        
        # Evaluate all models
        evaluation_results = self.ml_detector.evaluate_all_models(X_test, y_test)
        
        # Find best model
        best_model, best_score = self.ml_detector.get_best_model(evaluation_results)
        print(f"\nBest model: {best_model} (F1 Score: {best_score:.4f})")
        
        return evaluation_results, best_model
    
    def run_full_pipeline(self):
        """
        Chạy toàn bộ detection pipeline
        """
        print("🚀 Starting Full Detection Pipeline")
        print("=" * 50)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_features': self.config.MAX_FEATURES,
                'ngram_range': self.config.NGRAM_RANGE,
                'cv_folds': self.config.CV_FOLDS
            }
        }
        
        try:
            # 1. Load data
            data = self.load_data()
            
            # 2. Extract features
            train_features, test_features = self.extract_features(data)
            
            # 3. Rule-based evaluation
            rule_evaluation, rule_results = self.evaluate_rule_based(data)
            results['rule_based'] = rule_evaluation
            
            # 4. ML-based training and evaluation
            ml_evaluations, best_model = self.train_and_evaluate_ml(
                data, train_features, test_features
            )
            results['ml_based'] = ml_evaluations
            results['best_model'] = best_model
            
            # 5. Save models
            self.ml_detector.save_models(self.config.MODELS_DIR)
            
            # 6. Generate comparison report
            self.generate_comparison_report(results)
            
            print("\n✅ Pipeline completed successfully!")
            return results
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            results['error'] = str(e)
            return results
    
    def generate_comparison_report(self, results):
        """
        Tạo báo cáo so sánh các methods
        """
        print("\n=== COMPARISON REPORT ===")
        
        # Rule-based vs ML comparison
        rule_f1 = results['rule_based']['f1_score']
        
        print(f"Rule-based F1 Score: {rule_f1:.4f}")
        print("\nML Models Performance:")
        
        ml_results = []
        for model_name, evaluation in results['ml_based'].items():
            f1_score = evaluation['f1_score']
            ml_results.append({
                'model': model_name,
                'f1_score': f1_score,
                'precision': evaluation['precision'],
                'recall': evaluation['recall']
            })
            print(f"  {model_name}: F1={f1_score:.4f}")
        
        # Save detailed results
        with open(self.config.RESULTS_DIR / 'detection_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to {self.config.RESULTS_DIR / 'detection_results.json'}")

# Main execution
if __name__ == "__main__":
    pipeline = DetectionPipeline()
    results = pipeline.run_full_pipeline()