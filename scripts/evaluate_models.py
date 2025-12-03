#!/usr/bin/env python3
"""
Evaluate Detection System Models
Benchmark all ML and DL models on the SAME dataset for fair comparison
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from detection_system.config import Config as DetectionConfig
from detection_system.features.text_features.text_features import TextFeaturesExtractor


def evaluate_all_models_same_dataset():
    """Evaluate all ML and DL models on the same dataset for fair comparison"""
    
    print("=" * 70)
    print("DETECTION SYSTEM - UNIFIED MODEL EVALUATION")
    print("All models evaluated on the SAME test dataset")
    print("=" * 70)
    
    # Load HuggingFace test dataset (largest, most representative)
    test_path = project_root / "datasets" / "huggingface_test_20251113_050346.csv"
    print(f"\nLoading test dataset: {test_path.name}")
    
    df = pd.read_csv(test_path)
    X_test = df['prompt'].tolist()
    y_test = (df['label'] == 'malicious').astype(int).tolist()
    
    print(f"Total samples: {len(df)}")
    print(f"Malicious: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"Benign: {len(y_test) - sum(y_test)} ({(len(y_test) - sum(y_test))/len(y_test)*100:.1f}%)")
    
    results = []
    
    # ========================================
    # EVALUATE ML MODELS
    # ========================================
    print("\n" + "=" * 70)
    print("ML MODELS EVALUATION")
    print("=" * 70)
    
    # Extract features for ML models
    print("\nExtracting features...")
    feature_extractor = TextFeaturesExtractor(DetectionConfig)
    features_dict = feature_extractor.extract_all_features(X_test)
    
    # Combine features
    statistical = features_dict['statistical_features']
    tfidf = features_dict['tfidf_features']
    if hasattr(tfidf, 'toarray'):
        tfidf = tfidf.toarray()
    features = np.hstack([statistical, tfidf])
    print(f"Feature shape: {features.shape}")
    
    # Get saved models
    models_dir = project_root / "detection_system" / "saved_models"
    ml_models = ['logistic_regression', 'svm', 'gradient_boosting', 'random_forest', 'naive_bayes', 'svm_fast']
    
    print("\nEvaluating ML models...")
    for model_name in ml_models:
        model_path = models_dir / f"{model_name}.joblib"
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                y_pred = model.predict(features)
                
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                
                results.append({
                    'model': model_name,
                    'type': 'ML',
                    'accuracy': acc,
                    'f1': f1,
                    'precision': prec,
                    'recall': rec
                })
                
                print(f"  {model_name}: F1={f1:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
                
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
        else:
            print(f"  {model_name}: Model file not found")
    
    # ========================================
    # EVALUATE DEEP LEARNING MODEL
    # ========================================
    print("\n" + "=" * 70)
    print("DEEP LEARNING MODEL EVALUATION")
    print("=" * 70)
    
    try:
        from detection_system.models.deep_learning.transformer_detector import DeepLearningTrainer
        
        dl_model_path = models_dir / "deep_learning"
        if (dl_model_path / "model.pth").exists():
            print("\nLoading DistilBERT model...")
            dl_detector = DeepLearningTrainer()
            dl_detector.load_model(dl_model_path)
            
            print("Running predictions (this may take a while)...")
            y_pred_dl = dl_detector.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred_dl)
            f1 = f1_score(y_test, y_pred_dl)
            prec = precision_score(y_test, y_pred_dl)
            rec = recall_score(y_test, y_pred_dl)
            
            results.append({
                'model': 'distilbert',
                'type': 'DL',
                'accuracy': acc,
                'f1': f1,
                'precision': prec,
                'recall': rec
            })
            
            print(f"  distilbert: F1={f1:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
        else:
            print("  DistilBERT model not found")
    except Exception as e:
        print(f"  DistilBERT: Error - {e}")
    
    # ========================================
    # SUMMARY - SORTED BY F1 SCORE
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON (sorted by F1-Score)")
    print("=" * 70)
    
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Model':<22} {'Type':<6} {'F1':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 80)
    
    for i, r in enumerate(results_sorted, 1):
        print(f"{i:<6} {r['model']:<22} {r['type']:<6} {r['f1']:>10.4f} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")
    
    # Best model
    if results_sorted:
        best = results_sorted[0]
        print("\n" + "=" * 70)
        print(f"BEST MODEL: {best['model']} ({best['type']})")
        print(f"  F1-Score:  {best['f1']:.4f}")
        print(f"  Accuracy:  {best['accuracy']:.4f}")
        print(f"  Precision: {best['precision']:.4f}")
        print(f"  Recall:    {best['recall']:.4f}")
        print("=" * 70)
    
    return results


if __name__ == "__main__":
    evaluate_all_models_same_dataset()
