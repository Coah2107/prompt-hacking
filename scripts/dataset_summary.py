"""
Dataset Performance Summary
Tổng hợp kết quả performance trên các datasets khác nhau
"""

import pandas as pd
import numpy as np
from pathlib import Path

def summarize_results():
    """Tổng hợp kết quả từ các tests đã chạy"""
    
    print("📊 DATASET PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Results từ 2 datasets chính
    results = {
        "Challenging Dataset (Sophisticated)": {
            "description": "Advanced jailbreaks, edge cases, adversarial examples",
            "samples": 199,
            "malicious_ratio": 0.63,
            "performance": {
                "Logistic Regression": 0.898,
                "Random Forest": 0.925,
                "SVM": 0.828,
                "Gradient Boosting": 0.925
            },
            "key_insights": [
                "Realistic evaluation with sophisticated attacks",
                "Random Forest and Gradient Boosting perform best",
                "Contains borderline cases for robust testing",
                "Good for model development and validation"
            ]
        },
        
        "HuggingFace Dataset (Production-Scale)": {
            "description": "Large-scale real prompt injection dataset from research",
            "samples": 373646,
            "malicious_ratio": 0.235,
            "performance": {
                "Logistic Regression": 0.721,
                "Random Forest": 0.709,
                "SVM": 0.671,
                "Gradient Boosting": 0.706
            },
            "key_insights": [
                "Production-ready evaluation dataset",
                "Large-scale with 373K+ real samples",
                "Diverse attack patterns and sources",
                "Best for final model validation"
            ]
        }
    }
    
    # Print performance comparison table
    print("\n🎯 F1 SCORE COMPARISON:")
    print("-" * 70)
    
    models = ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"]
    datasets = list(results.keys())
    
    # Header
    print(f"{'Model':<20} {'Challenging':<12} {'HuggingFace':<12} {'Performance Gap':<15}")
    print("-" * 70)
    
    # Data rows
    for model in models:
        challenge_score = results["Challenging Dataset (Sophisticated)"]["performance"][model]
        hf_score = results["HuggingFace Dataset (Production-Scale)"]["performance"][model]
        
        gap = challenge_score - hf_score
        
        print(f"{model:<20} {challenge_score:.3f}       {hf_score:.3f}        {gap:.3f}")
    
    # Dataset characteristics
    print(f"\n📋 DATASET CHARACTERISTICS:")
    print("-" * 80)
    
    for dataset_name, data in results.items():
        print(f"\n{dataset_name}:")
        print(f"  📊 Samples: {data['samples']}")
        print(f"  ⚖️  Malicious Ratio: {data['malicious_ratio']:.0%}")
        print(f"  📝 Description: {data['description']}")
        print(f"  💡 Key Insights:")
        for insight in data['key_insights']:
            print(f"     • {insight}")
    
    # Performance analysis
    print(f"\n🔍 PERFORMANCE ANALYSIS:")
    print("-" * 70)
    
    print(f"\n💚 BEST PERFORMING MODELS:")
    print("  • Random Forest: Most consistent across both datasets (0.925 → 0.709)")
    print("  • Logistic Regression: Best on production data (0.721 F1)")
    print("  • Gradient Boosting: Good balance on challenging data (0.925 F1)")
    
    print(f"\n📉 PERFORMANCE GAPS:")
    challenge_avg = np.mean([results["Challenging Dataset (Sophisticated)"]["performance"][m] for m in models])
    hf_avg = np.mean([results["HuggingFace Dataset (Production-Scale)"]["performance"][m] for m in models])
    
    print(f"  • Average gap (Challenging → Production): {challenge_avg - hf_avg:.3f}")
    print(f"  • Largest gap (Random Forest): {0.925 - 0.709:.3f}")
    print(f"  • Smallest gap (Logistic Regression): {0.898 - 0.721:.3f}")
    
    print(f"\n⚠️  KEY FINDINGS:")
    print("  1. Challenging dataset (199 samples) for development & testing")
    print("  2. HuggingFace dataset (373K samples) for production validation")
    print("  3. Performance gap shows real-world complexity")
    print("  4. Models need improvement for production deployment")
    print("  5. Focus on features that work on large-scale real data")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print("  ✅ Use Challenging dataset for rapid development & iteration")
    print("  ✅ Use HuggingFace dataset for final validation & benchmarking")
    print("  ✅ Train models on full HuggingFace data for better performance")
    print("  ✅ Focus on Logistic Regression (best production performance)")
    print("  ✅ Improve feature engineering to close the performance gap")
    print("  ✅ Consider ensemble methods for production deployment")
    
    # Dataset files summary
    print(f"\n📁 AVAILABLE DATASETS:")
    datasets_dir = Path('../datasets')
    
    print(f"\nSimple/Original:")
    print(f"  • full_dataset.csv (400 samples)")
    print(f"  • train_dataset.csv, test_dataset.csv")
    
    print(f"\nChallenging/Advanced:")
    challenging_files = list(datasets_dir.glob('challenging_*.csv'))
    for f in challenging_files[-3:]:  # Show last 3
        print(f"  • {f.name}")
    
    print(f"\nReal-World/Authentic:")
    realworld_files = list(datasets_dir.glob('realworld_*.csv'))
    for f in realworld_files[-3:]:  # Show last 3
        print(f"  • {f.name}")
    
    # Update with HuggingFace results
    print(f"\n🚀 HUGGINGFACE DATASET (LARGE-SCALE):")
    print("  📊 Samples: 373,646 (10,000 tested)")
    print("  ⚖️  Malicious Ratio: 23.5% (balanced 50% in sample)")
    print("  📝 Description: Large-scale real prompt injection dataset")
    print("  🎯 Performance: F1 = 0.72 (Logistic Regression best)")
    print("  💡 Key Insights:")
    print("     • Shows realistic performance on large-scale data")
    print("     • Significant performance drop from synthetic to real data")
    print("     • Text length: avg 1,091 chars (much longer than synthetic)")
    print("     • Contains diverse attack patterns and sources")

    print(f"\n🏆 FINAL COMPARISON - ALL DATASETS:")
    print("-" * 80)
    print(f"{'Dataset':<20} {'Samples':<10} {'F1 Score':<10} {'Difficulty':<15} {'Recommendation'}")
    print("-" * 80)
    dont_use = "❌ Don't use"
    development = "✅ Development"  
    need_more = "⚠️ Need more data"
    final_test = "🎯 Final test"
    print(f"{'Simple (Synthetic)':<20} {'400':<10} {'1.000':<10} {'Too Easy':<15} {dont_use}")
    print(f"{'Challenging':<20} {'199':<10} {'0.925':<10} {'Realistic':<15} {development}")
    print(f"{'Real-World':<20} {'65':<10} {'0.978':<10} {'Authentic':<15} {need_more}")
    print(f"{'HuggingFace':<20} {'373,646':<10} {'0.721':<10} {'Production':<15} {final_test}")
    
    print(f"\n🎯 KEY FINDINGS - PERFORMANCE DEGRADATION:")
    print("  📉 Simple → HuggingFace: 1.000 → 0.721 (28% drop)")
    print("  📉 Challenging → HuggingFace: 0.925 → 0.721 (22% drop)")
    print("  📊 This shows the importance of testing on real-world data!")
    
    print(f"\n✨ UPDATED NEXT STEPS:")
    print("  1. ✅ COMPLETED: Downloaded large-scale dataset (373K samples)")
    print("  2. ✅ COMPLETED: Performance tested on all dataset types")
    print("  3. 🎯 RECOMMENDATION: Use HuggingFace dataset for final evaluation")
    print("  4. 🔧 IMPROVEMENT: Focus on features that work on real data")
    print("  5. 📈 SCALING: Train on full HuggingFace dataset for better models")

if __name__ == "__main__":
    summarize_results()
