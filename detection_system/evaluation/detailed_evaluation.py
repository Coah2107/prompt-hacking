"""
Detailed Evaluation và Visualization
Lý do: Deep analysis kết quả để understand strengths/weaknesses của từng approach
"""

import pandas as pd
impor        print(f"\nRECOMMENDATIONS:")
        if best_f1 >= 0.8:
            print("   Excellent performance - ready for production")
        elif best_f1 >= 0.6:
            print("   Good performance - consider fine-tuning")
        else:
            print("   Needs improvement - more data or feature engineering required") as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import json

class DetailedEvaluator:
    def __init__(self, results_path):
        with open(results_path, 'r') as f:
            self.results = json.load(f)
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices cho tất cả models
        """
        # Rule-based confusion matrix
        rule_cm = np.array(self.results['rule_based']['confusion_matrix'])
        
        # ML models confusion matrices
        ml_models = list(self.results['ml_based'].keys())
        
        # Create subplots
        n_models = len(ml_models) + 1  # +1 for rule-based
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot rule-based
        sns.heatmap(rule_cm, annot=True, fmt='d', ax=axes[0], 
                   xticklabels=['Benign', 'Malicious'], 
                   yticklabels=['Benign', 'Malicious'])
        axes[0].set_title('Rule-based Detector')
        
        # Plot ML models
        for i, model_name in enumerate(ml_models[:5]):  # Max 5 models
            if i + 1 < len(axes):
                cm = np.array(self.results['ml_based'][model_name]['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[i+1],
                           xticklabels=['Benign', 'Malicious'], 
                           yticklabels=['Benign', 'Malicious'])
                axes[i+1].set_title(f'{model_name.replace("_", " ").title()}')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_performance_comparison(self):
        """
        Tạo biểu đồ so sánh performance
        """
        # Collect metrics
        models = ['Rule-based']
        f1_scores = [self.results['rule_based']['f1_score']]
        precisions = [self.results['rule_based']['precision']]
        recalls = [self.results['rule_based']['recall']]
        
        for model_name, evaluation in self.results['ml_based'].items():
            models.append(model_name.replace('_', ' ').title())
            f1_scores.append(evaluation['f1_score'])
            precisions.append(evaluation['precision'])
            recalls.append(evaluation['recall'])
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Model': models,
            'F1 Score': f1_scores,
            'Precision': precisions,
            'Recall': recalls
        })
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # F1 Score
        bars1 = axes[0].bar(comparison_df['Model'], comparison_df['F1 Score'], color='skyblue')
        axes[0].set_title('F1 Score Comparison')
        axes[0].set_ylabel('F1 Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Precision
        bars2 = axes[1].bar(comparison_df['Model'], comparison_df['Precision'], color='lightgreen')
        axes[1].set_title('Precision Comparison')
        axes[1].set_ylabel('Precision')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Recall
        bars3 = axes[2].bar(comparison_df['Model'], comparison_df['Recall'], color='lightcoral')
        axes[2].set_title('Recall Comparison')
        axes[2].set_ylabel('Recall')
        axes[2].tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def generate_summary_report(self):
        """
        Tạo báo cáo tóm tắt
        """
        print("=" * 60)
        print("DETECTION SYSTEM EVALUATION SUMMARY")
        print("=" * 60)
        
        # Best performing model
        best_model = self.results['best_model']
        best_f1 = self.results['ml_based'][best_model]['f1_score']
        
        print(f"\nBEST PERFORMING MODEL: {best_model.upper()}")
        print(f"   F1 Score: {best_f1:.4f}")
        print(f"   Precision: {self.results['ml_based'][best_model]['precision']:.4f}")
        print(f"   Recall: {self.results['ml_based'][best_model]['recall']:.4f}")
        
        # Rule-based performance
        rule_f1 = self.results['rule_based']['f1_score']
        print(f"\nRULE-BASED DETECTOR:")
        print(f"   F1 Score: {rule_f1:.4f}")
        print(f"   Precision: {self.results['rule_based']['precision']:.4f}")
        print(f"   Recall: {self.results['rule_based']['recall']:.4f}")
        print(f"   Inference Time: {self.results['rule_based']['inference_time']:.2f}s")
        
        # Improvement analysis
        improvement = ((best_f1 - rule_f1) / rule_f1) * 100
        print(f"\nIMPROVEMENT:")
        print(f"   ML vs Rule-based: {improvement:+.1f}% F1 score improvement")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if best_f1 > 0.85:
            print("   Excellent performance - ready for production")
        elif best_f1 > 0.75:
            print("   Good performance - consider fine-tuning")
        else:
            print("   Needs improvement - more data or feature engineering required")

if __name__ == "__main__":
    evaluator = DetailedEvaluator('results/detection_results.json')
    
    # Generate all evaluations
    evaluator.plot_confusion_matrices()
    comparison_df = evaluator.create_performance_comparison()
    evaluator.generate_summary_report()
    
    # Save comparison data
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nEvaluation completed! Check results/ folder for outputs.")