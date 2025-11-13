"""
Project Path Utilities
Lý do: Centralized path management để tránh hardcoded paths và support multiple developers
"""

import os
from pathlib import Path

def get_project_root():
    """
    Get the absolute path to project root directory
    Works from any subdirectory in the project
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    
    # Go up directories until we find the project root (contains specific marker files)
    project_markers = ['README.md', 'datasets', 'detection_system', 'prevention_system']
    
    search_dir = current_dir
    max_levels = 5  # Prevent infinite loop
    
    for _ in range(max_levels):
        # Check if this directory contains project markers
        marker_count = 0
        for marker in project_markers:
            marker_path = os.path.join(search_dir, marker)
            if os.path.exists(marker_path):
                marker_count += 1
        
        # If we found most markers, this is likely the project root
        if marker_count >= 3:
            return os.path.abspath(search_dir)
        
        # Go up one level
        parent_dir = os.path.dirname(search_dir)
        if parent_dir == search_dir:  # Reached filesystem root
            break
        search_dir = parent_dir
    
    # Fallback: assume current directory is project root
    return os.path.abspath(current_dir)

def get_relative_path(*path_parts):
    """
    Get a path relative to project root
    
    Args:
        *path_parts: Path components to join (e.g., 'datasets', 'train.csv')
    
    Returns:
        Absolute path to the specified location
    """
    project_root = get_project_root()
    return os.path.join(project_root, *path_parts)

def ensure_directory_exists(path):
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        path: Directory path to check/create
    """
    os.makedirs(path, exist_ok=True)

def get_datasets_dir():
    """Get the datasets directory path"""
    return Path(get_relative_path('datasets'))

def get_results_dir():
    """Get the results directory path"""
    return Path(get_relative_path('results'))

def get_models_dir():
    """Get the saved models directory path"""
    return Path(get_relative_path('detection_system', 'saved_models'))

def get_prevention_logs_dir():
    """Get the prevention logs directory path"""
    return Path(get_relative_path('results', 'prevention_logs'))

def get_prevention_metrics_dir():
    """Get the prevention metrics directory path"""
    return Path(get_relative_path('results', 'prevention_metrics'))

# Pre-computed common paths
PROJECT_ROOT = get_project_root()
DATASETS_DIR = get_relative_path('datasets')
MODELS_DIR = get_relative_path('detection_system', 'saved_models')
RESULTS_DIR = get_relative_path('results')
PREVENTION_LOGS_DIR = get_relative_path('results', 'prevention_logs')
PREVENTION_METRICS_DIR = get_relative_path('results', 'prevention_metrics')

# Create common directories
ensure_directory_exists(RESULTS_DIR)
ensure_directory_exists(MODELS_DIR)
ensure_directory_exists(PREVENTION_LOGS_DIR)
ensure_directory_exists(PREVENTION_METRICS_DIR)

if __name__ == "__main__":
    # Test the path utilities
    print("🧪 Testing Project Path Utilities")
    print("=" * 50)
    
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Datasets Dir: {DATASETS_DIR}")
    print(f"Models Dir: {MODELS_DIR}")
    print(f"Results Dir: {RESULTS_DIR}")
    
    # Test relative path generation
    test_paths = [
        ('datasets', 'train.csv'),
        ('detection_system', 'config.py'),
        ('prevention_system', 'config.py'),
        ('results', 'test_report.json')
    ]
    
    print(f"\n📁 Testing relative paths:")
    for path_parts in test_paths:
        full_path = get_relative_path(*path_parts)
        exists = "✅" if os.path.exists(full_path) else "❌"
        print(f"  {exists} {'/'.join(path_parts)} -> {full_path}")
    
    print("\n✅ Path utilities test completed!")
