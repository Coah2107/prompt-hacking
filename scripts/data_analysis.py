import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

class DatasetAnalyzer:
    def __init__(self, dataset_path):
        self.df = pd.read_csv(dataset_path)
        plt.style.use('seaborn-v0_8')
        
    def basic_statistics(self):
        """Thống kê cơ bản"""
        print("=== THỐNG KÊ CƠ BẢN ===")
        print(f"Tổng số samples: {len(self.df)}")
        print(f"Số features: {len(self.df.columns)}")
        print(f"\nPhân bố label:")
        print(self.df['label'].value_counts())
        print(f"\nPhân bố attack type:")
        print(self.df['attack_type'].value_counts())
        
        # Thống kê độ dài prompt
        print(f"\n=== THỐNG KÊ ĐỘ DÀI ===")
        print(f"Độ dài trung bình: {self.df['length'].mean():.2f}")
        print(f"Độ dài min: {self.df['length'].min()}")
        print(f"Độ dài max: {self.df['length'].max()}")
        
    def visualize_data(self):
        """Tạo các biểu đồ visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Phân bố label
        self.df['label'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Phân bố Label')
        axes[0,0].set_xlabel('Label')
        axes[0,0].set_ylabel('Số lượng')
        
        # 2. Phân bố attack type
        attack_counts = self.df['attack_type'].value_counts()
        axes[0,1].pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Phân bố Attack Type')
        
        # 3. Histogram độ dài prompt
        self.df['length'].hist(bins=30, ax=axes[1,0])
        axes[1,0].set_title('Phân bố độ dài Prompt')
        axes[1,0].set_xlabel('Độ dài (ký tự)')
        axes[1,0].set_ylabel('Tần suất')
        
        # 4. Boxplot độ dài theo label
        sns.boxplot(data=self.df, x='label', y='length', ax=axes[1,1])
        axes[1,1].set_title('Độ dài Prompt theo Label')
        
        plt.tight_layout()
        plt.savefig('results/dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_text_patterns(self):
        """Phân tích các pattern trong text"""
        print("\n=== PHÂN TÍCH TEXT PATTERNS ===")
        
        # Từ khóa xuất hiện nhiều trong malicious prompts
        malicious_prompts = self.df[self.df['label'] == 'malicious']['prompt']
        all_malicious_text = ' '.join(malicious_prompts).lower()
        
        # Đếm từ
        words = all_malicious_text.split()
        common_words = Counter(words).most_common(20)
        
        print("Top 20 từ khóa trong malicious prompts:")
        for word, count in common_words:
            print(f"  {word}: {count}")
            
    def export_summary(self):
        """Xuất báo cáo tóm tắt"""
        summary = {
            "total_samples": len(self.df),
            "malicious_samples": len(self.df[self.df['label'] == 'malicious']),
            "benign_samples": len(self.df[self.df['label'] == 'benign']),
            "attack_types": self.df['attack_type'].value_counts().to_dict(),
            "avg_length": self.df['length'].mean(),
            "avg_word_count": self.df['word_count'].mean()
        }
        
        with open('results/dataset_summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
            
        print("\nĐã xuất báo cáo summary vào results/dataset_summary.json")

if __name__ == "__main__":
    analyzer = DatasetAnalyzer('datasets/full_dataset.csv')
    analyzer.basic_statistics()
    analyzer.visualize_data()
    analyzer.analyze_text_patterns()
    analyzer.export_summary()