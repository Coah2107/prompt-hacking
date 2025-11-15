"""
Traditional ML Models for Prompt Hacking Detection
Lý do: Proven effective cho text classification, fast training và inference
"""

import pandas as pd
import numpy as np
import time
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
from pathlib import Path

class TraditionalMLDetector:
    def __init__(self, config=None):
        # Default config if none provided
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        self.models = self._initialize_models()
        self.trained_models = {}
        self.feature_extractor = None
        
    def _get_default_config(self):
        """Provide default configuration if none given"""
        class DefaultConfig:
            RANDOM_STATE = 42
            CV_FOLDS = 5
            PRIMARY_METRIC = 'f1_score'
            MAX_FEATURES = 5000
            MIN_DF = 2
            MAX_DF = 0.95
            NGRAM_RANGE = (1, 2)
        
        return DefaultConfig()
        
    def _initialize_models(self):
        """
        Khởi tạo các ML models
        Lý do: Different algorithms có strengths khác nhau cho text classification
        """
        models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.config.RANDOM_STATE,
                    max_iter=1000,
                    C=1.0  # Regularization parameter
                ),
                'description': 'Linear model, good baseline, interpretable'
            },
            
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=50,   # Reduced from 100 for faster training
                    random_state=self.config.RANDOM_STATE,
                    max_depth=15,      # Limited depth for speed
                    min_samples_split=5,  # Increased for speed
                    min_samples_leaf=2,   # Increased for speed
                    class_weight='balanced',  # Handle imbalanced data
                    n_jobs=-1          # Use all CPU cores
                ),
                'description': 'Fast ensemble method, handles non-linear patterns well'
            },
            
            'svm': {
                'model': LinearSVC(
                    C=0.1,  # Lower C for faster training (less strict)
                    random_state=self.config.RANDOM_STATE,
                    class_weight='balanced',
                    tol=1e-2,  # Relaxed tolerance for faster convergence
                    max_iter=2000,  # Limited iterations for speed
                    dual=False,  # Use primal formulation (faster for n_samples > n_features)
                    loss='squared_hinge'  # Faster loss function
                ),
                'description': 'Ultra-fast Linear SVM - 10x faster than SVC (no probability support)'
            },
            
            'svm_fast': {
                'model': SGDClassifier(
                    loss='hinge',      # SVM loss function
                    alpha=0.01,        # Regularization strength
                    random_state=self.config.RANDOM_STATE,
                    max_iter=1000,     # Fast convergence
                    tol=1e-3,         # Tolerance for stopping
                    class_weight='balanced',
                    learning_rate='optimal',  # Automatic learning rate
                    n_jobs=-1         # Use all CPU cores
                ),
                'description': 'Ultra-fast SGD SVM - 10-100x faster than regular SVM'
            },
            
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=50,   # Reduced from 100 for speed
                    learning_rate=0.2, # Increased for faster convergence
                    max_depth=5,       # Limited depth
                    random_state=self.config.RANDOM_STATE,
                    subsample=0.8      # Use subset for faster training
                ),
                'description': 'Fast sequential ensemble with good performance'
            },
            
            'naive_bayes': {
                'model': MultinomialNB(
                    alpha=1.0  # Smoothing parameter
                ),
                'description': 'Fast, works well with sparse features like TF-IDF'
            }
        }
        
        return models
    
    def prepare_features(self, X_statistical, X_tfidf):
        """
        Chuẩn bị features cho training
        Lý do: Combine statistical và TF-IDF features
        """
        from scipy.sparse import hstack, csr_matrix
        
        # Handle negative values in statistical features (MultinomialNB requirement)
        X_statistical_positive = np.maximum(X_statistical, 0)  # Replace negative with 0
        
        # Convert statistical features to sparse matrix
        X_statistical_sparse = csr_matrix(X_statistical_positive)
        
        # Combine features
        X_combined = hstack([X_statistical_sparse, X_tfidf])
        
        # Final check: ensure no negative values for MultinomialNB
        if hasattr(X_combined, 'data'):
            X_combined.data = np.maximum(X_combined.data, 0)
        else:
            X_combined = np.maximum(X_combined.toarray(), 0)
            X_combined = csr_matrix(X_combined)
        
        return X_combined
    
    def train_single_model(self, model_name, X_train, y_train, X_val=None, y_val=None, progress_callback=None):
        """
        Train một model cụ thể với progress tracking
        Lý do: Modular approach, có thể train từng model riêng biệt
        """
        import sys
        
        model_config = self.models[model_name]
        model = model_config['model']
        
        # Train model với progress indication
        if not progress_callback:
            print(f"🔄 Training {model_name}...")
            print(f"   📊 Data: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features")
        
        # Progress updates every 10%
        def update_progress(progress, stage):
            if progress_callback:
                progress_callback(stage, progress)
                # Small delay to see progress updates
                import time
                time.sleep(0.05)  # 50ms delay để nhìn thấy progress
        
        # Stage 1: Fitting model (0-50%)
        update_progress(0, "Fitting")
        update_progress(10, "Fitting")
        update_progress(20, "Fitting")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        update_progress(30, "Fitting")
        update_progress(40, "Fitting")
        update_progress(50, "Fitting")
        
        if not progress_callback:
            print(f"   ⚡ Fitting model... ✅ ({fit_time:.1f}s)")
        
        # Stage 2: Cross-validation (50-100%)
        update_progress(50, "CV")
        update_progress(60, "CV")
        
        cv_start = time.time()
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=self.config.CV_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE),
            scoring='f1'
        )
        cv_time = time.time() - cv_start
        
        update_progress(70, "CV")
        update_progress(80, "CV")
        update_progress(90, "CV")
        
        if not progress_callback:
            print(f"   🔍 Cross-validation ({self.config.CV_FOLDS} folds)... ✅ ({cv_time:.1f}s)")
            print(f"   📈 CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Validation evaluation if provided
        if X_val is not None and y_val is not None:
            update_progress(95, "Validation")
            val_predictions = model.predict(X_val)
            val_probabilities = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            val_report = classification_report(y_val, val_predictions, output_dict=True)
            if not progress_callback:
                print(f"Validation F1 score: {val_report['weighted avg']['f1-score']:.4f}")
        
        # Final progress
        update_progress(100, "Done")
        
        # Store trained model
        model_result = {
            'model': model,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        self.trained_models[model_name] = model_result
        
        return model
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train tất cả models
        Lý do: So sánh performance của different algorithms
        """
        print("Starting training for all models...")
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                model = self.train_single_model(model_name, X_train, y_train, X_val, y_val)
                results[model_name] = self.trained_models[model_name]
                print(f"✅ {model_name} trained successfully")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Đánh giá một model cụ thể
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]['model']
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        evaluation = {
            'model_name': model_name,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'auc_score': auc_score,
            'f1_score': report['weighted avg']['f1-score'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall']
        }
        
        return evaluation
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Đánh giá tất cả trained models
        """
        results = {}
        
        for model_name in self.trained_models.keys():
            if 'error' not in self.trained_models[model_name]:
                evaluation = self.evaluate_model(model_name, X_test, y_test)
                results[model_name] = evaluation
                
                print(f"\n=== {model_name} Results ===")
                print(f"F1 Score: {evaluation['f1_score']:.4f}")
                print(f"Precision: {evaluation['precision']:.4f}")
                print(f"Recall: {evaluation['recall']:.4f}")
                if evaluation['auc_score']:
                    print(f"AUC Score: {evaluation['auc_score']:.4f}")
        
        return results
    
    def get_best_model(self, results):
        """
        Tìm model tốt nhất dựa trên primary metric
        """
        best_model = None
        best_score = 0
        
        for model_name, result in results.items():
            if 'error' not in result:
                score = result[self.config.PRIMARY_METRIC]
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model, best_score
    
    def save_models(self, save_dir):
        """
        Lưu trained models
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_data in self.trained_models.items():
            if 'error' not in model_data:
                try:
                    # Ensure model_data is a dict and has 'model' key
                    if isinstance(model_data, dict) and 'model' in model_data:
                        model_path = save_dir / f"{model_name}.joblib"
                        joblib.dump(model_data['model'], model_path)
                        print(f"Saved {model_name} to {model_path}")
                    else:
                        print(f"⚠️  Warning: {model_name} data structure invalid: {type(model_data)}")
                except Exception as e:
                    print(f"❌ Error saving {model_name}: {str(e)}")
    
    def load_model(self, model_path):
        """
        Load pre-trained model
        """
        return joblib.load(model_path)
    
    def detect_single_prompt(self, prompt, model_name='logistic_regression'):
        """
        Detect if a single prompt is malicious using specified model
        """
        # Use path utils for consistent path management
        import os
        try:
            from utils.path_utils import get_models_dir
            models_dir = get_models_dir()
            model_path = models_dir / f'{model_name}.joblib'
        except ImportError:
            # Fallback to relative path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, '..', '..', 'saved_models')
            model_path = os.path.join(models_dir, f'{model_name}.joblib')
        
        if not os.path.exists(model_path):
            return {
                'prompt': prompt,
                'prediction': 'unknown',
                'confidence': 0.0,
                'error': f'Model {model_name} not found at {model_path}'
            }
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Extract features (simplified for testing)
            from detection_system.features.text_features.text_features import TextFeaturesExtractor
            extractor = TextFeaturesExtractor(self.config)
            features = extractor.extract_all_features([prompt])
            
            # Prepare features
            X = self.prepare_features(
                features['statistical_features'], 
                features['tfidf_features']
            )
            
            # Predict
            prediction = model.predict(X)[0]
            confidence = model.predict_proba(X)[0].max() if hasattr(model, 'predict_proba') else prediction
            
            return {
                'prompt': prompt,
                'prediction': 'malicious' if prediction == 1 else 'benign',
                'confidence': float(confidence),
                'model_used': model_name
            }
            
        except Exception as e:
            return {
                'prompt': prompt,
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }

# Test function
if __name__ == "__main__":
    import sys
    import os
    
    try:
        # Use absolute imports if available
        from detection_system.config import Config
        from detection_system.features.text_features.text_features import TextFeaturesExtractor
        from utils.path_utils import get_datasets_dir
    except ImportError:
        # Fallback to old method for backward compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        detection_system_dir = os.path.join(current_dir, '../..')
        sys.path.insert(0, detection_system_dir)
        
        from config import Config
        
        # Add path for text_features module
        features_dir = os.path.join(detection_system_dir, 'features', 'text_features')
        sys.path.insert(0, features_dir)
    
    from text_features import TextFeaturesExtractor
    
    # Create test config with lower min_df for small dataset
    class TestConfig:
        MAX_FEATURES = 1000
        MIN_DF = 1  # Lower for testing with few samples
        MAX_DF = 0.95
        NGRAM_RANGE = (1, 2)
        RANDOM_STATE = 42
        CV_FOLDS = 3
        PRIMARY_METRIC = 'f1_score'
    
    # Load real dataset for testing
    print("Loading real dataset...")
    
    # Path to dataset - try current datasets first
    try:
        datasets_dir = get_datasets_dir()
        
        # Look for challenging dataset first
        challenging_files = list(datasets_dir.glob('challenging_dataset_*.csv'))
        if challenging_files:
            dataset_path = max(challenging_files, key=lambda x: x.stat().st_mtime)
            print(f"📊 Using challenging dataset: {dataset_path.name}")
        else:
            # Fallback to old path
            dataset_path = datasets_dir / 'full_dataset.csv'
    except (ImportError, NameError):
        # Fallback for backward compatibility
        dataset_path = os.path.join(detection_system_dir, '..', 'datasets', 'full_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure you have datasets available in datasets/ directory")
        exit(1)
    
    # Load dataset
    import pandas as pd
    df = pd.read_csv(dataset_path)
    print(f"📊 Dataset loaded: {df.shape[0]} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Prepare data
    sample_texts = df['prompt'].tolist()
    sample_labels = (df['label'] == 'malicious').astype(int).tolist()  # Convert to binary
    
    print(f"Label distribution:")
    print(f"  Malicious: {sum(sample_labels)}")
    print(f"  Benign: {len(sample_labels) - sum(sample_labels)}")
    
    # Analyze dataset characteristics
    print(f"\n🔍 DATASET ANALYSIS:")
    malicious_texts = [text for text, label in zip(sample_texts, sample_labels) if label == 1]
    benign_texts = [text for text, label in zip(sample_texts, sample_labels) if label == 0]
    
    # Check for obvious patterns
    suspicious_keywords = ['ignore', 'dan', 'jailbreak', 'override', 'bypass', 'act as', 'pretend']
    
    malicious_with_keywords = 0
    benign_with_keywords = 0
    
    for text in malicious_texts:
        if any(keyword in text.lower() for keyword in suspicious_keywords):
            malicious_with_keywords += 1
            
    for text in benign_texts:
        if any(keyword in text.lower() for keyword in suspicious_keywords):
            benign_with_keywords += 1
    
    print(f"Malicious texts with obvious keywords: {malicious_with_keywords}/{len(malicious_texts)} ({malicious_with_keywords/len(malicious_texts)*100:.1f}%)")
    print(f"Benign texts with suspicious keywords: {benign_with_keywords}/{len(benign_texts)} ({benign_with_keywords/len(benign_texts)*100:.1f}%)")
    
    # Show examples
    print(f"\nExample malicious prompts:")
    for i in range(min(3, len(malicious_texts))):
        print(f"  {i+1}. \"{malicious_texts[i][:80]}...\"")
        
    print(f"\nExample benign prompts:")
    for i in range(min(3, len(benign_texts))):
        print(f"  {i+1}. \"{benign_texts[i][:80]}...\"")
    
    if malicious_with_keywords > len(malicious_texts) * 0.8:
        print(f"\n⚠️  WARNING: {malicious_with_keywords/len(malicious_texts)*100:.1f}% of malicious prompts contain obvious keywords!")
        print("This makes classification trivially easy and unrealistic.")
    
    # Extract features using TestConfig
    extractor = TextFeaturesExtractor(TestConfig)
    features = extractor.extract_all_features(sample_texts)
    
    # Prepare features
    detector = TraditionalMLDetector(TestConfig)
    X = detector.prepare_features(
        features['statistical_features'], 
        features['tfidf_features']
    )
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Proper train/test split for real dataset
    print(f"\nSplitting dataset: {len(sample_texts)} samples total")
    
    from sklearn.model_selection import train_test_split
    
    try:
        # Split dataset into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, sample_labels, 
            test_size=0.2, 
            random_state=TestConfig.RANDOM_STATE,
            stratify=sample_labels  # Ensure balanced splits
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train all models
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        training_results = detector.train_all_models(X_train, y_train)
        
        # Evaluate all models
        print("\n" + "="*60)
        print("EVALUATING MODELS")
        print("="*60)
        
        evaluation_results = detector.evaluate_all_models(X_test, y_test)
        
        # Summary table
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'F1 Score':<10} {'Precision':<12} {'Recall':<10} {'AUC':<8}")
        print("-" * 80)
        
        for model_name, result in evaluation_results.items():
            print(f"{model_name:<20} {result['f1_score']:<10.4f} {result['precision']:<12.4f} "
                  f"{result['recall']:<10.4f} {result['auc_score']:<8.4f}")
        
        # Find best model
        if evaluation_results:
            best_model_name, best_score = detector.get_best_model(evaluation_results)
            
            print(f"\n🏆 BEST MODEL: {best_model_name}")
            print(f"F1 Score: {best_score:.4f}")
            print(f"AUC Score: {evaluation_results[best_model_name]['auc_score']:.4f}")
            
            # Show confusion matrix for best model
            cm = evaluation_results[best_model_name]['confusion_matrix']
            print(f"\nConfusion Matrix for {best_model_name}:")
            print(f"              Predicted")
            print(f"           Benign  Malicious")
            print(f"Actual Benign    {cm[0][0]:3d}      {cm[0][1]:3d}")
            print(f"    Malicious    {cm[1][0]:3d}      {cm[1][1]:3d}")
            
            # Analyze feature importance (for interpretable models)
            if best_model_name in ['logistic_regression', 'random_forest']:
                print(f"\n🔍 ANALYZING PERFECT RESULTS:")
                print("This may indicate:")
                print("  1. Dataset is too simple/synthetic")
                print("  2. Features perfectly separate classes") 
                print("  3. Possible data leakage")
                print("  4. Need more challenging/realistic data")
                
                # Show feature statistics
                model = detector.trained_models[best_model_name]['model']
                if hasattr(model, 'coef_'):
                    coef = model.coef_[0]
                    top_features = np.argsort(np.abs(coef))[-10:][::-1]
                    print(f"\nTop 10 most important features:")
                    for i, feat_idx in enumerate(top_features):
                        print(f"  {i+1}. Feature {feat_idx}: {coef[feat_idx]:.4f}")
                        
            # Show some predictions from best model
            best_model = detector.trained_models[best_model_name]['model']
            
            # Check probability distributions  
            print(f"\n📊 PROBABILITY ANALYSIS:")
            try:
                # Get prediction probabilities
                probabilities = best_model.predict_proba(X_test)[:, 1]  # Probability of malicious class
                
                high_conf_malicious = np.sum(probabilities > 0.9)
                high_conf_benign = np.sum(probabilities < 0.1) 
                uncertain = np.sum((probabilities >= 0.3) & (probabilities <= 0.7))
                
                print(f"High confidence malicious (>0.9): {high_conf_malicious}/{len(probabilities)}")
                print(f"High confidence benign (<0.1): {high_conf_benign}/{len(probabilities)}")
                print(f"Uncertain predictions (0.3-0.7): {uncertain}/{len(probabilities)}")
                
                if uncertain == 0:
                    print("⚠️  WARNING: No uncertain predictions - dataset may be too simple!")
                    
                # Check for extreme probabilities (signs of overfitting)
                extreme_probs = np.sum((probabilities < 0.01) | (probabilities > 0.99))
                print(f"Extreme probabilities (<0.01 or >0.99): {extreme_probs}/{len(probabilities)} ({extreme_probs/len(probabilities)*100:.1f}%)")
                
            except Exception as e:
                print(f"❌ Error analyzing probabilities: {e}")
            
            # Analyze dataset characteristics
            print(f"\n🔍 DATASET QUALITY ANALYSIS:")
            
            try:
                # Convert sparse matrix to dense if needed
                if hasattr(X_test, 'toarray'):
                    X_test_dense = X_test.toarray()
                else:
                    X_test_dense = X_test
                
                # Check feature separability
                perfect_features = 0
                for i in range(X_test_dense.shape[1]):
                    feature_col = X_test_dense[:, i]
                    malicious_vals = feature_col[y_test == 1]
                    benign_vals = feature_col[y_test == 0]
                    
                    if malicious_vals.shape[0] > 0 and benign_vals.shape[0] > 0:
                        mal_max, mal_min = np.max(malicious_vals), np.min(malicious_vals)
                        ben_max, ben_min = np.max(benign_vals), np.min(benign_vals)
                        
                        # Check if ranges don't overlap (perfect separation)
                        if mal_max < ben_min or ben_max < mal_min:
                            perfect_features += 1
                
                print(f"Features with perfect class separation: {perfect_features}/{X_test_dense.shape[1]}")
                if perfect_features > 10:
                    print("⚠️  Many features perfectly separate classes - dataset may be too synthetic!")
                    
            except Exception as e:
                print(f"❌ Error analyzing feature separation: {e}")
            
            # Text pattern analysis  
            print(f"\n📝 TEXT PATTERN ANALYSIS:")
            # Load dataset again for analysis
            import pandas as pd
            df = pd.read_csv(dataset_path)
            malicious_samples = df[df['label'] == 'malicious']
            benign_samples = df[df['label'] == 'benign']
            
            # Length analysis
            mal_lengths = [len(text) for text in malicious_samples['prompt']]
            ben_lengths = [len(text) for text in benign_samples['prompt']] 
            
            print(f"Average text length:")
            print(f"  Malicious: {np.mean(mal_lengths):.1f} ± {np.std(mal_lengths):.1f}")
            print(f"  Benign: {np.mean(ben_lengths):.1f} ± {np.std(ben_lengths):.1f}")
            
            # Pattern analysis
            mal_questions = sum(1 for text in malicious_samples['prompt'] if '?' in text)
            ben_questions = sum(1 for text in benign_samples['prompt'] if '?' in text)
            
            print(f"Texts containing '?':")
            print(f"  Malicious: {mal_questions}/{len(malicious_samples)} ({mal_questions/len(malicious_samples)*100:.1f}%)")
            print(f"  Benign: {ben_questions}/{len(benign_samples)} ({ben_questions/len(benign_samples)*100:.1f}%)")
            
            # Check for obvious trigger words
            trigger_words = ['hack', 'illegal', 'bypass', 'jailbreak', 'manipulate', 'exploit']
            mal_with_triggers = sum(1 for text in malicious_samples['prompt'] 
                                  if any(word in text.lower() for word in trigger_words))
            ben_with_triggers = sum(1 for text in benign_samples['prompt'] 
                                  if any(word in text.lower() for word in trigger_words))
            
            print(f"Texts with trigger words {trigger_words}:")
            print(f"  Malicious: {mal_with_triggers}/{len(malicious_samples)} ({mal_with_triggers/len(malicious_samples)*100:.1f}%)")
            print(f"  Benign: {ben_with_triggers}/{len(benign_samples)} ({ben_with_triggers/len(benign_samples)*100:.1f}%)")
            
            if mal_with_triggers > len(malicious_samples) * 0.5:
                print("⚠️  Over 50% of malicious samples contain obvious trigger words!")
            
            # Convert sparse matrix to dense for SVM if needed
            X_test_pred = X_test
            if best_model_name == 'svm' and hasattr(X_test, 'toarray'):
                X_test_pred = X_test.toarray()
            
            predictions = best_model.predict(X_test_pred)
            probabilities = best_model.predict_proba(X_test_pred)[:, 1] if hasattr(best_model, 'predict_proba') else predictions
            
            # Show a few example predictions
            print(f"\nSample predictions from {best_model_name}:")
            test_indices = np.random.choice(len(y_test), min(5, len(y_test)), replace=False)
            
            for i in test_indices:
                text_idx = X_test.getrow(i) if hasattr(X_test, 'getrow') else i
                original_idx = i  # This is approximate since we don't track original indices
                
                status = "✅" if predictions[i] == y_test[i] else "❌"
                prob = probabilities[i] if hasattr(probabilities, '__len__') else probabilities
                
                print(f"{status} Pred: {predictions[i]} (prob: {prob:.3f}, actual: {y_test[i]})")
                
                # Try to show text (approximation)
                if i < len(sample_texts):
                    print(f"   \"{sample_texts[i][:80]}...\"")
            
            # Save models
            models_dir = os.path.join(detection_system_dir, 'saved_models')
            detector.save_models(models_dir)
            print(f"\n💾 Models saved to: {models_dir}")
        
        else:
            print("❌ No models were successfully trained and evaluated")
            
    except Exception as e:
        print(f"❌ Error during training/evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n✅ Traditional ML detector test completed!")