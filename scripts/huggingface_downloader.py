"""
HuggingFace Dataset Downloader
Download và xử lý dataset 'ahsanayub/malicious-prompts' từ HuggingFace
"""

import pandas as pd
from datasets import load_dataset
import json
from datetime import datetime
from pathlib import Path
import re

class HuggingFaceDatasetDownloader:
    def __init__(self):
        self.dataset_name = "ahsanayub/malicious-prompts"
        self.raw_data = None
        self.processed_data = []
        
    def download_dataset(self):
        """Download dataset từ HuggingFace"""
        print(f"📥 Downloading dataset: {self.dataset_name}")
        
        try:
            # Load dataset
            dataset = load_dataset(self.dataset_name)
            
            # Check available splits
            print(f"Available splits: {list(dataset.keys())}")
            
            # Get the main split (usually 'train' or the first available)
            if 'train' in dataset:
                self.raw_data = dataset['train']
            else:
                split_name = list(dataset.keys())[0]
                self.raw_data = dataset[split_name]
                
            print(f"SUCCESS Dataset downloaded successfully!")
            print(f"ANALYSIS Total samples: {len(self.raw_data)}")
            
            # Show column names
            if hasattr(self.raw_data, 'column_names'):
                print(f"📋 Columns: {self.raw_data.column_names}")
            
            # Show first few examples
            print(f"\nPERFORMANCE First 3 examples:")
            for i in range(min(3, len(self.raw_data))):
                example = self.raw_data[i]
                print(f"  {i+1}. {example}")
                
            return True
            
        except Exception as e:
            print(f"ERROR Error downloading dataset: {str(e)}")
            return False
    
    def explore_dataset_structure(self):
        """Khám phá cấu trúc của dataset"""
        if self.raw_data is None:
            print("ERROR No data to explore. Download dataset first.")
            return
            
        print(f"\nPERFORMANCE EXPLORING DATASET STRUCTURE")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset size: {len(self.raw_data)}")
        print(f"Features: {self.raw_data.features}")
        
        # Convert to pandas for easier analysis
        df = self.raw_data.to_pandas()
        
        print(f"\nANALYSIS COLUMN ANALYSIS:")
        for col in df.columns:
            print(f"\n{col}:")
            if df[col].dtype == 'object':
                # Show sample values for text columns
                sample_values = df[col].dropna().head(5).tolist()
                print(f"  Sample values: {sample_values}")
                
                # Check for unique values (if reasonable count)
                unique_count = df[col].nunique()
                print(f"  Unique values: {unique_count}")
                
                if unique_count < 20:
                    print(f"  Value counts: {df[col].value_counts().to_dict()}")
            else:
                # Numeric column stats
                print(f"  Type: {df[col].dtype}")
                print(f"  Range: {df[col].min()} - {df[col].max()}")
                
        print(f"\n📋 DATA SAMPLE:")
        print(df.head().to_string(max_colwidth=100))
        
        return df
    
    def process_huggingface_dataset(self, df):
        """Xử lý và chuẩn hóa dataset theo format của project"""
        print(f"\n🔄 PROCESSING DATASET")
        print("=" * 40)
        
        processed_samples = []
        
        # Analyze column names to determine mapping
        columns = df.columns.tolist()
        print(f"Available columns: {columns}")
        
        # Common column name mappings
        text_columns = ['prompt', 'text', 'input', 'message', 'question', 'query']
        label_columns = ['label', 'category', 'type', 'class', 'is_malicious', 'malicious']
        
        # Find text column
        text_col = None
        for col in text_columns:
            if col in columns:
                text_col = col
                break
        
        # Find label column  
        label_col = None
        for col in label_columns:
            if col in columns:
                label_col = col
                break
                
        print(f"Detected text column: {text_col}")
        print(f"Detected label column: {label_col}")
        
        if text_col is None:
            print("WARNING  Could not detect text column. Using first text-like column.")
            text_col = columns[0]
            
        # Process each sample
        for idx, row in df.iterrows():
            try:
                # Extract text
                text = str(row[text_col]) if text_col else ""
                
                # Extract and normalize label
                if label_col:
                    raw_label = row[label_col]
                    # Normalize label to malicious/benign
                    label = self.normalize_label(raw_label)
                else:
                    # If no label column, try to infer from other columns or text
                    label = self.infer_label(row, text)
                
                # Determine attack type and severity
                attack_type, severity = self.classify_attack_type(text)
                
                # Create processed sample
                sample = {
                    "id": f"hf_{idx:04d}",
                    "prompt": text,
                    "label": label,
                    "attack_type": attack_type,
                    "severity": severity,
                    "length": len(text),
                    "word_count": len(text.split()),
                    "source": "huggingface_ahsanayub",
                    "original_data": dict(row),  # Keep original for reference
                    "created_at": datetime.now().isoformat()
                }
                
                processed_samples.append(sample)
                
            except Exception as e:
                print(f"WARNING  Error processing sample {idx}: {str(e)}")
                continue
        
        self.processed_data = processed_samples
        print(f"SUCCESS Processed {len(processed_samples)} samples")
        
        return processed_samples
    
    def normalize_label(self, raw_label):
        """Chuẩn hóa label thành malicious/benign"""
        if pd.isna(raw_label):
            return "unknown"
            
        label_str = str(raw_label).lower().strip()
        
        # Malicious indicators
        malicious_keywords = [
            'malicious', 'harmful', 'toxic', 'bad', 'attack', 'jailbreak', 
            'injection', 'exploit', 'threat', 'dangerous', 'inappropriate',
            '1', 'true', 'yes', 'positive'
        ]
        
        # Benign indicators  
        benign_keywords = [
            'benign', 'safe', 'good', 'normal', 'legitimate', 'clean',
            '0', 'false', 'no', 'negative'
        ]
        
        # Check for malicious
        for keyword in malicious_keywords:
            if keyword in label_str:
                return "malicious"
                
        # Check for benign
        for keyword in benign_keywords:
            if keyword in label_str:
                return "benign"
                
        # Default fallback
        return "unknown"
    
    def infer_label(self, row, text):
        """Suy luận label từ text hoặc các cột khác"""
        text_lower = text.lower()
        
        # Check for obvious malicious patterns
        malicious_patterns = [
            'ignore previous', 'jailbreak', 'bypass', 'hack', 
            'pretend you are', 'act as', 'roleplay', 'dan mode',
            'developer mode', 'unrestricted', 'override'
        ]
        
        for pattern in malicious_patterns:
            if pattern in text_lower:
                return "malicious"
                
        # Check other columns for hints
        for col, val in row.items():
            if pd.isna(val):
                continue
            val_str = str(val).lower()
            if any(word in val_str for word in ['malicious', 'attack', 'harmful', 'jailbreak']):
                return "malicious"
            elif any(word in val_str for word in ['benign', 'safe', 'legitimate']):
                return "benign"
        
        # Default to unknown if can't determine
        return "unknown"
    
    def classify_attack_type(self, text):
        """Phân loại attack type và severity"""
        text_lower = text.lower()
        
        # Attack type classification
        if any(pattern in text_lower for pattern in ['ignore previous', 'ignore all', 'disregard']):
            attack_type = "prompt_injection"
            severity = "high"
        elif any(pattern in text_lower for pattern in ['jailbreak', 'dan', 'developer mode', 'unrestricted']):
            attack_type = "jailbreaking" 
            severity = "high"
        elif any(pattern in text_lower for pattern in ['pretend', 'act as', 'roleplay', 'imagine']):
            attack_type = "social_engineering"
            severity = "medium"
        elif any(pattern in text_lower for pattern in ['bypass', 'override', 'circumvent']):
            attack_type = "system_manipulation"
            severity = "high"
        else:
            attack_type = "none"
            severity = "none"
            
        return attack_type, severity
    
    def save_processed_dataset(self, processed_data, prefix="huggingface"):
        """Lưu processed dataset"""
        if not processed_data:
            print("ERROR No processed data to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Statistics
        print(f"\nANALYSIS PROCESSED DATASET STATISTICS:")
        print(f"Total samples: {len(df)}")
        print(f"Label distribution:")
        print(df['label'].value_counts())
        print(f"Attack type distribution:")
        print(df['attack_type'].value_counts())
        
        # Save files
        datasets_dir = Path('../datasets')
        datasets_dir.mkdir(exist_ok=True)
        
        csv_path = datasets_dir / f'{prefix}_dataset_{timestamp}.csv'
        json_path = datasets_dir / f'{prefix}_dataset_{timestamp}.json'
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        df.to_json(json_path, orient='records', indent=2)
        
        # Create train/test split
        from sklearn.model_selection import train_test_split
        
        # Only split if we have enough samples and both classes
        if len(df) >= 10 and len(df['label'].unique()) > 1:
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, 
                stratify=df['label'] if df['label'].nunique() > 1 else None
            )
            
            train_path = datasets_dir / f'{prefix}_train_{timestamp}.csv'
            test_path = datasets_dir / f'{prefix}_test_{timestamp}.csv'
            
            train_df.to_csv(train_path, index=False, encoding='utf-8')
            test_df.to_csv(test_path, index=False, encoding='utf-8')
            
            print(f"\n💾 Dataset saved:")
            print(f"  Full: {csv_path} ({len(df)} samples)")
            print(f"  Train: {train_path} ({len(train_df)} samples)")
            print(f"  Test: {test_path} ({len(test_df)} samples)")
        else:
            print(f"\n💾 Dataset saved:")
            print(f"  Full: {csv_path} ({len(df)} samples)")
        
        # Create metadata
        metadata = {
            "source": "huggingface:ahsanayub/malicious-prompts",
            "download_date": datetime.now().isoformat(),
            "total_samples": len(df),
            "label_distribution": df['label'].value_counts().to_dict(),
            "attack_type_distribution": df['attack_type'].value_counts().to_dict(),
            "processing_notes": "Downloaded from HuggingFace and processed for prompt security detection"
        }
        
        metadata_path = datasets_dir / f'{prefix}_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        print(f"  Metadata: {metadata_path}")
        
        return csv_path, train_path if 'train_path' in locals() else None, test_path if 'test_path' in locals() else None
    
    def run_complete_pipeline(self):
        """Chạy toàn bộ pipeline download và xử lý"""
        print("HUGGINGFACE HUGGINGFACE DATASET DOWNLOAD PIPELINE")
        print("=" * 60)
        
        # Step 1: Download
        if not self.download_dataset():
            return False
            
        # Step 2: Explore structure
        df = self.explore_dataset_structure()
        if df is None:
            return False
            
        # Step 3: Process data
        processed_data = self.process_huggingface_dataset(df)
        if not processed_data:
            return False
            
        # Step 4: Save processed dataset
        paths = self.save_processed_dataset(processed_data)
        
        print(f"\nSUCCESS Pipeline completed successfully!")
        print(f"TARGET Dataset from HuggingFace is now ready for testing your models")
        
        return True

def main():
    """Main function"""
    downloader = HuggingFaceDatasetDownloader()
    success = downloader.run_complete_pipeline()
    
    if success:
        print(f"\nREPORT NEXT STEPS:")
        print("  1. Check the datasets/ folder for new files")
        print("  2. Run: python3 dataset_benchmark.py (to compare with other datasets)")
        print("  3. Test your models on this new dataset")
        print("  4. Update your model training with this additional data")
    
    return success

if __name__ == "__main__":
    main()
