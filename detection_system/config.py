"""
Configuration file cho Detection System
Lý do: Centralize tất cả settings, dễ dàng thay đổi parameters mà không cần sửa code
"""

import os
from pathlib import Path

class Config:
    # Đường dẫn files
    BASE_DIR = Path(__file__).parent.parent
    DATASET_DIR = BASE_DIR / "datasets"
    MODELS_DIR = BASE_DIR / "detection_system" / "saved_models"
    RESULTS_DIR = BASE_DIR / "results"
    
    # Dataset files
    TRAIN_DATA = DATASET_DIR / "train_dataset.csv"
    TEST_DATA = DATASET_DIR / "test_dataset.csv"
    FULL_DATA = DATASET_DIR / "full_dataset.csv"
    
    # Model parameters
    RANDOM_STATE = 42  # Để reproducible results
    TEST_SIZE = 0.2
    CV_FOLDS = 5  # Cross-validation folds
    
    # Feature extraction parameters
    MAX_FEATURES = 10000  # Số lượng features tối đa cho TF-IDF
    MIN_DF = 2  # Minimum document frequency
    MAX_DF = 0.95  # Maximum document frequency
    NGRAM_RANGE = (1, 3)  # Unigrams, bigrams, trigrams
    
    # Model thresholds
    CLASSIFICATION_THRESHOLD = 0.5
    
    # Evaluation metrics
    PRIMARY_METRIC = "f1_score"
    
    # Tạo directories nếu chưa tồn tại
    @classmethod
    def create_dirs(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize
Config.create_dirs()