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
    
    # Dataset files - Updated to use HuggingFace datasets
    TRAIN_DATA = DATASET_DIR / "huggingface_train_20251113_050346.csv"
    TEST_DATA = DATASET_DIR / "huggingface_test_20251113_050346.csv"
    FULL_DATA = DATASET_DIR / "huggingface_dataset_20251113_050346.csv"
    
    # Model parameters - Optimized for faster training
    RANDOM_STATE = 42  # Để reproducible results
    TEST_SIZE = 0.2
    CV_FOLDS = 3  # Reduced from 5 to 3 for faster training
    
    # Feature extraction parameters - Optimized for speed
    MAX_FEATURES = 5000  # Reduced from 10000 for faster training
    MIN_DF = 5  # Increased to reduce feature space
    MAX_DF = 0.9  # Slightly more restrictive
    NGRAM_RANGE = (1, 2)  # Reduced from (1,3) to (1,2) for speed
    
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