"""
Configuration for Prevention System
Lý do: Centralized config cho tất cả prevention components với các thresholds và rules
"""

from pathlib import Path
# import yaml  # Commented out to avoid dependency issues

class PreventionConfig:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    PREVENTION_DIR = BASE_DIR / "prevention_system"
    
    # Detection system integration
    DETECTION_SYSTEM_DIR = BASE_DIR / "detection_system"
    
    # Logs and monitoring
    LOGS_DIR = BASE_DIR / "results" / "prevention_logs"
    METRICS_DIR = BASE_DIR / "results" / "prevention_metrics"
    
    # Input filtering thresholds
    INPUT_FILTER_CONFIG = {
        'max_prompt_length': 2000,  # Maximum characters allowed
        'max_word_count': 400,      # Maximum words allowed
        'min_prompt_length': 1,     # Minimum characters required
        'blocked_patterns': [       # Immediate block patterns
            r'ignore\s+(?:all\s+)?previous\s+instructions',
            r'act\s+as\s+dan',
            r'jailbreak',
            r'override\s+(?:all\s+)?(?:safety|security)',
        ],
        'suspicious_threshold': 0.7,  # Threshold for suspicious content
        'malicious_threshold': 0.9    # Threshold for immediate blocking
    }
    
    # Output validation config
    OUTPUT_VALIDATION_CONFIG = {
        'max_response_length': 5000,
        'blocked_topics': [
            'violence', 'illegal_activities', 'harmful_substances',
            'privacy_violation', 'hate_speech', 'misinformation'
        ],
        'safety_check_threshold': 0.8,
        'content_policy_threshold': 0.9
    }
    
    # Real-time monitoring config
    MONITORING_CONFIG = {
        'rate_limiting': {
            'max_requests_per_minute': 60,
            'max_requests_per_hour': 1000,
            'suspicious_pattern_limit': 5  # Max suspicious attempts before blocking
        },
        'alerting': {
            'attack_threshold': 0.95,      # Alert on high-confidence attacks
            'anomaly_threshold': 3.0,      # Standard deviations for anomaly detection
            'email_notifications': True,
            'slack_notifications': False
        },
        'logging': {
            'log_level': 'INFO',
            'log_rotation': '1 day',
            'max_log_size': '100MB',
            'retention_days': 30
        }
    }
    
    # Adaptive learning config
    ADAPTIVE_CONFIG = {
        'learning_enabled': True,
        'update_frequency': 'daily',    # How often to update patterns
        'min_samples_for_update': 100,  # Minimum samples needed for pattern update
        'confidence_threshold': 0.85,   # Confidence needed to add new patterns
        'false_positive_threshold': 0.1 # Max acceptable false positive rate
    }
    
    # API configuration
    API_CONFIG = {
        'port': 8000,
        'host': '0.0.0.0',
        'workers': 4,
        'timeout': 30,
        'cors_enabled': True,
        'rate_limiting_enabled': True,
        'authentication_required': True
    }
    
    @classmethod
    def create_dirs(cls):
        """Tạo các directories cần thiết"""
        directories = [
            cls.LOGS_DIR,
            cls.METRICS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("Created prevention system directories")
    
    @classmethod
    def load_custom_config(cls, config_path):
        """Load custom configuration từ YAML file"""
        try:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
            
            # Update configs with custom values
            for key, value in custom_config.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
            
            print(f"Loaded custom config from {config_path}")
        except FileNotFoundError:
            print(f"Custom config file not found: {config_path}")
        except Exception as e:
            print(f"Error loading custom config: {e}")

# Aliases for compatibility
Config = PreventionConfig

# Initialize
PreventionConfig.create_dirs()