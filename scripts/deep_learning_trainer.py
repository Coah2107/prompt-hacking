"""
Deep Learning Detection Integration
Tích hợp deep learning model vào detection system

Author: AI Security Team  
Date: November 2024
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import numpy as np
import pandas as pd
from detection_system.models.deep_learning.transformer_detector import DeepLearningTrainer
from utils.path_utils import get_project_root, get_datasets_dir
import time
import json

class DeepLearningDetector:
    """
    Deep Learning Detector cho production use
    """
    
    def __init__(self, model_path=None):
        self.trainer = DeepLearningTrainer()
        self.model_loaded = False
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.trainer.load_model(model_path)
            self.model_loaded = True
            print(f"✅ Deep learning model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.model_loaded = False
    
    def predict(self, text):
        """
        Predict single text
        Returns: dict with prediction and confidence
        """
        if not self.model_loaded:
            return {
                'prediction': 'benign',  # Safe default
                'confidence': 0.0,
                'method': 'fallback'
            }
        
        try:
            # Get probabilities
            probs = self.trainer.predict_proba([text])
            prediction = self.trainer.predict([text])
            
            # Format result
            benign_prob = probs[0][0]
            malicious_prob = probs[0][1]
            pred_label = 'malicious' if prediction[0] == 1 else 'benign'
            confidence = max(benign_prob, malicious_prob)
            
            return {
                'prediction': pred_label,
                'confidence': float(confidence),
                'malicious_score': float(malicious_prob),
                'benign_score': float(benign_prob),
                'method': 'deep_learning'
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'prediction': 'benign',  # Safe default
                'confidence': 0.0,
                'method': 'error_fallback'
            }
    
    def batch_predict(self, texts):
        """
        Predict multiple texts
        """
        if not self.model_loaded:
            return [{'prediction': 'benign', 'confidence': 0.0, 'method': 'fallback'} 
                   for _ in texts]
        
        try:
            predictions = self.trainer.predict(texts)
            probabilities = self.trainer.predict_proba(texts)
            
            results = []
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                benign_prob = probs[0]
                malicious_prob = probs[1]
                pred_label = 'malicious' if pred == 1 else 'benign'
                confidence = max(benign_prob, malicious_prob)
                
                results.append({
                    'prediction': pred_label,
                    'confidence': float(confidence),
                    'malicious_score': float(malicious_prob),
                    'benign_score': float(benign_prob),
                    'method': 'deep_learning'
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return [{'prediction': 'benign', 'confidence': 0.0, 'method': 'error_fallback'} 
                   for _ in texts]

def train_new_model():
    """
    Train a new deep learning model
    """
    print("🚀 TRAINING NEW DEEP LEARNING MODEL")
    print("=" * 50)
    
    try:
        # Import and run training
        from detection_system.models.deep_learning.transformer_detector import main as train_main
        trainer, best_f1 = train_main()
        
        print(f"✅ Training completed with F1: {best_f1:.4f}")
        return trainer, best_f1
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def test_deep_learning_model(num_samples=20):
    """
    Test deep learning model performance
    """
    print("🧪 TESTING DEEP LEARNING MODEL")
    print("=" * 50)
    
    # Load model
    models_dir = get_project_root() / "detection_system" / "saved_models" / "deep_learning"
    
    if not models_dir.exists():
        print("❌ No deep learning model found. Training new model...")
        trainer, f1_score = train_new_model()
        if trainer is None:
            return
    
    # Initialize detector
    detector = DeepLearningDetector(models_dir)
    
    if not detector.model_loaded:
        print("❌ Failed to load model for testing")
        return
    
    # Load test data
    datasets_dir = get_datasets_dir()
    
    # Try different datasets
    test_files = list(datasets_dir.glob("*test*.csv"))
    if not test_files:
        # Use full datasets and sample
        all_files = list(datasets_dir.glob("huggingface_dataset_*.csv"))
        if not all_files:
            all_files = list(datasets_dir.glob("challenging_dataset_*.csv"))
        
        if all_files:
            test_files = [all_files[0]]
    
    if not test_files:
        print("❌ No test datasets found")
        return
    
    # Load and sample data
    df = pd.read_csv(test_files[0])
    
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    
    print(f"📊 Testing on {len(df)} samples from {test_files[0].name}")
    
    # Convert labels
    if 'label' in df.columns:
        if df['label'].dtype == 'object':
            df['binary_label'] = (df['label'] == 'malicious').astype(int)
        else:
            df['binary_label'] = df['label']
    else:
        print("❌ No label column found")
        return
    
    # Make predictions
    start_time = time.time()
    
    texts = df['prompt'].tolist()
    results = detector.batch_predict(texts)
    
    processing_time = time.time() - start_time
    
    # Evaluate results
    predictions = [1 if r['prediction'] == 'malicious' else 0 for r in results]
    true_labels = df['binary_label'].tolist()
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    print(f"\n📈 DEEP LEARNING PERFORMANCE")
    print(f"⏱️  Processing time: {processing_time:.2f}s ({processing_time/len(texts)*1000:.1f}ms per sample)")
    print(f"🎯 Accuracy: {accuracy:.1%}")
    print(f"📊 F1 Score: {f1:.4f}")
    
    # Detailed report
    report = classification_report(true_labels, predictions, 
                                 target_names=['benign', 'malicious'],
                                 digits=4)
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print(f"\n🔍 CONFUSION MATRIX:")
    print(f"              Predicted")
    print(f"           Benign  Malicious")
    print(f"Actual Benign    {cm[0][0]:3d}     {cm[0][1]:3d}")
    print(f"    Malicious    {cm[1][0]:3d}     {cm[1][1]:3d}")
    
    # Sample predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    for i in range(min(10, len(results))):
        text = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
        result = results[i]
        true_label = 'malicious' if true_labels[i] == 1 else 'benign'
        
        status = "✅" if result['prediction'] == true_label else "❌"
        
        print(f"{status} '{text}'")
        print(f"   True: {true_label} | Pred: {result['prediction']} | Confidence: {result['confidence']:.3f}")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'processing_time': processing_time,
        'samples_tested': len(df),
        'detailed_results': results
    }

def benchmark_comparison():
    """
    Compare deep learning vs traditional methods
    """
    print("📊 BENCHMARK COMPARISON: Deep Learning vs Traditional ML")
    print("=" * 70)
    
    # Test deep learning
    print("🤖 Testing Deep Learning Model...")
    dl_results = test_deep_learning_model(num_samples=50)
    
    # Test traditional ML (if available)
    print("\n🔬 Testing Traditional ML Model...")
    try:
        from detection_system.detector_pipeline import MLDetectorPipeline
        
        # Load traditional detector
        traditional_detector = MLDetectorPipeline()
        
        # Load test data (same as deep learning)
        datasets_dir = get_datasets_dir()
        test_files = list(datasets_dir.glob("*test*.csv"))
        
        if not test_files:
            all_files = list(datasets_dir.glob("huggingface_dataset_*.csv"))
            if not all_files:
                all_files = list(datasets_dir.glob("challenging_dataset_*.csv"))
            if all_files:
                test_files = [all_files[0]]
        
        if test_files:
            df = pd.read_csv(test_files[0])
            if len(df) > 50:
                df = df.sample(n=50, random_state=42)
            
            # Test traditional ML
            start_time = time.time()
            ml_results = traditional_detector.predict_batch(df['prompt'].tolist())
            ml_time = time.time() - start_time
            
            # Calculate traditional ML metrics
            ml_predictions = [1 if r.get('is_malicious', False) else 0 for r in ml_results]
            true_labels = (df['label'] == 'malicious').astype(int).tolist()
            
            from sklearn.metrics import f1_score, accuracy_score
            ml_accuracy = accuracy_score(true_labels, ml_predictions)
            ml_f1 = f1_score(true_labels, ml_predictions)
            
            print(f"📈 Traditional ML Performance:")
            print(f"   Accuracy: {ml_accuracy:.1%}")
            print(f"   F1 Score: {ml_f1:.4f}")
            print(f"   Processing Time: {ml_time:.2f}s")
            
        else:
            print("❌ No test data available for traditional ML")
            ml_results = None
            
    except Exception as e:
        print(f"❌ Traditional ML test failed: {e}")
        ml_results = None
    
    # Comparison summary
    if dl_results and ml_results is not None:
        print(f"\n🏆 COMPARISON SUMMARY:")
        print(f"{'Metric':<20} {'Deep Learning':<15} {'Traditional ML':<15} {'Winner':<10}")
        print("-" * 65)
        
        dl_acc = dl_results['accuracy']
        dl_f1 = dl_results['f1_score']
        dl_time = dl_results['processing_time']
        
        acc_winner = "Deep Learning" if dl_acc > ml_accuracy else "Traditional ML"
        f1_winner = "Deep Learning" if dl_f1 > ml_f1 else "Traditional ML"
        speed_winner = "Deep Learning" if dl_time < ml_time else "Traditional ML"
        
        print(f"{'Accuracy':<20} {dl_acc:<15.1%} {ml_accuracy:<15.1%} {acc_winner:<10}")
        print(f"{'F1 Score':<20} {dl_f1:<15.4f} {ml_f1:<15.4f} {f1_winner:<10}")
        print(f"{'Speed (seconds)':<20} {dl_time:<15.2f} {ml_time:<15.2f} {speed_winner:<10}")

def main():
    """
    Main function with options
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Learning Prompt Hacking Detection")
    parser.add_argument('command', choices=['train', 'test', 'benchmark'], 
                       help='Command to execute')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of samples for testing (default: 20)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_new_model()
    elif args.command == 'test':
        test_deep_learning_model(args.samples)
    elif args.command == 'benchmark':
        benchmark_comparison()

if __name__ == "__main__":
    main()
