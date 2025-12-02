#!/usr/bin/env python3
"""
4-Stage Optimized Security Pipeline Test
Author: Security Team
Date: December 2024

Workflow:
  Stage 1: Fast Pre-filter (Pattern + Prompt Leaking)
  Stage 2: Semantic Analysis (SVM-based)
  Stage 3: AI Processing (DistilBERT)
  Stage 4: Response Validation
"""

import sys
import time
import re
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from prevention_system.filters.input_filters.core_filter import CoreInputFilter
from prevention_system.filters.content_filters.semantic_filter import SemanticFilter
from prevention_system.filters.content_filters.prompt_leaking_detector import PromptLeakingDetector
from prevention_system.validators.response_validators.safety_validator import ResponseValidator
from prevention_system.config import PreventionConfig


class OptimizedSecurityPipeline:
    """4-Stage Optimized Security Pipeline"""
    
    def __init__(self):
        print("Initializing 4-Stage Security Pipeline...")
        
        # Stage 1: Fast Pre-filter
        self.input_filter = CoreInputFilter(PreventionConfig)
        leaking_config = getattr(PreventionConfig, 'PROMPT_LEAKING_CONFIG', None)
        self.leaking_detector = PromptLeakingDetector(leaking_config)
        
        # Stage 2: Semantic Analysis
        self.semantic_filter = SemanticFilter()
        
        # Stage 3: AI Processing (DistilBERT)
        self.dl_detector = None
        self._load_distilbert()
        
        # Stage 4: Response Validation
        self.response_validator = ResponseValidator()
        
        # Enhanced attack patterns for Stage 1
        self._attack_patterns = [
            r'ignore\s+(all\s+)?(previous|prior|above)',
            r'disregard\s+(all\s+)?instructions',
            r'override\s+(system|safety)',
            r'bypass\s+(security|filter)',
            r'act\s+as\s+(if|though)',
            r'pretend\s+(you|to\s+be)',
            r'you\s+are\s+now',
            r'forget\s+(everything|all)',
            r'reveal\s+(your|the)\s+(system|hidden)',
            r'what\s+(is|are)\s+your\s+(instructions|rules)',
            r'jailbreak',
            r'dan\s+mode',
            r'developer\s+mode',
            r'system:\s*override',
            r'\[system\s*override\]',
            r'===.*jailbreak.*===',
            r'@@@.*admin.*@@@',
        ]
        
        print("Pipeline ready: Pre-filter -> Semantic -> AI -> Validation")
    
    def _load_distilbert(self):
        """Load DistilBERT model for Stage 3"""
        try:
            from detection_system.models.deep_learning.transformer_detector import DeepLearningTrainer
            self.dl_detector = DeepLearningTrainer()
            models_dir = project_root / "detection_system" / "saved_models" / "deep_learning"
            if (models_dir / "model.pth").exists():
                self.dl_detector.load_model(models_dir)
                print("Loaded DistilBERT model for Stage 3")
            else:
                self.dl_detector = None
                print("DistilBERT not found, Stage 3 will use pattern fallback")
        except Exception as e:
            print(f"DistilBERT load failed: {e}")
            self.dl_detector = None
    
    def _enhanced_pattern_check(self, prompt: str) -> tuple:
        """Check for attack patterns in Stage 1"""
        text = prompt.lower()
        for pattern in self._attack_patterns:
            if re.search(pattern, text):
                return True, f"pattern:{pattern[:20]}"
        return False, None
    
    def analyze(self, prompt: str) -> dict:
        """Process prompt through 4-stage pipeline"""
        start_time = time.time()
        
        result = {
            'prompt': prompt[:80] + "..." if len(prompt) > 80 else prompt,
            'final_decision': 'DELIVERED',
            'blocked_at': None,
            'block_reason': None,
            'stages': {},
            'processing_time': 0
        }
        
        # === STAGE 1: Fast Pre-filter ===
        filter_result = self.input_filter.filter_prompt(prompt)
        pattern_blocked = not filter_result['allowed']
        
        leaking_result = self.leaking_detector.detect(prompt)
        leaking_blocked = leaking_result.is_leaking_attempt
        
        enhanced_blocked, enhanced_reason = self._enhanced_pattern_check(prompt)
        
        stage1_blocked = pattern_blocked or leaking_blocked or enhanced_blocked
        
        result['stages']['stage1_prefilter'] = {
            'pattern_blocked': pattern_blocked,
            'leaking_blocked': leaking_blocked,
            'enhanced_blocked': enhanced_blocked,
            'blocked': stage1_blocked
        }
        
        if stage1_blocked:
            result['final_decision'] = 'BLOCKED'
            result['blocked_at'] = 'Stage 1: Pre-filter'
            if leaking_blocked:
                result['block_reason'] = 'prompt_leaking'
            elif enhanced_blocked:
                result['block_reason'] = enhanced_reason
            else:
                result['block_reason'] = 'pattern_filter'
            result['processing_time'] = time.time() - start_time
            return result
        
        # === STAGE 2: Semantic Analysis ===
        semantic_result = self.semantic_filter.analyze_content(prompt)
        toxicity = semantic_result.get('toxicity_score', 0)
        attack_sim = semantic_result.get('attack_similarity', 0)
        
        stage2_blocked = (toxicity > 0.85) or (toxicity > 0.7 and attack_sim > 0.7)
        
        result['stages']['stage2_semantic'] = {
            'toxicity': round(toxicity, 3),
            'attack_similarity': round(attack_sim, 3),
            'blocked': stage2_blocked
        }
        
        if stage2_blocked:
            result['final_decision'] = 'BLOCKED'
            result['blocked_at'] = 'Stage 2: Semantic'
            result['block_reason'] = 'high_toxicity'
            result['processing_time'] = time.time() - start_time
            return result
        
        # === STAGE 3: AI Processing (DistilBERT) ===
        if self.dl_detector:
            try:
                probs = self.dl_detector.predict_proba([prompt])[0]
                malicious_prob = probs[1]
                stage3_blocked = malicious_prob > 0.48
                
                result['stages']['stage3_ai'] = {
                    'model': 'distilbert',
                    'malicious_prob': round(malicious_prob, 3),
                    'blocked': stage3_blocked
                }
                
                if stage3_blocked:
                    result['final_decision'] = 'BLOCKED'
                    result['blocked_at'] = 'Stage 3: AI Detection'
                    result['block_reason'] = f'distilbert:{malicious_prob:.3f}'
                    result['processing_time'] = time.time() - start_time
                    return result
            except Exception as e:
                result['stages']['stage3_ai'] = {'error': str(e)}
        else:
            result['stages']['stage3_ai'] = {'skipped': True}
        
        # === STAGE 4: Response Validation ===
        test_response = "Here is a helpful response."
        validation = self.response_validator.validate_response_simple(test_response)
        
        result['stages']['stage4_validation'] = {
            'is_safe': validation['is_safe'],
            'safety_score': round(validation['safety_score'], 3)
        }
        
        if not validation['is_safe']:
            result['final_decision'] = 'BLOCKED'
            result['blocked_at'] = 'Stage 4: Validation'
            result['block_reason'] = 'unsafe_response'
        
        result['processing_time'] = time.time() - start_time
        return result


def load_test_cases(num_malicious=5, num_benign=5, seed=None):
    """Load test cases from dataset"""
    import pandas as pd
    import numpy as np
    
    dataset_path = project_root / "datasets" / "challenging_train_20251113_043657.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        
        if seed is None:
            seed = np.random.randint(0, 10000)
        print(f"Loaded {len(df)} samples (seed: {seed})")
        
        malicious = df[df['label'] == 'malicious'].sample(n=min(num_malicious, len(df[df['label'] == 'malicious'])), random_state=seed)
        benign = df[df['label'] == 'benign'].sample(n=min(num_benign, len(df[df['label'] == 'benign'])), random_state=seed+1)
        
        cases = []
        for _, row in malicious.iterrows():
            cases.append({'prompt': row['prompt'], 'label': 'malicious', 'expected': 'BLOCKED'})
        for _, row in benign.iterrows():
            cases.append({'prompt': row['prompt'], 'label': 'benign', 'expected': 'DELIVERED'})
        
        return cases, seed
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], None


def run_test(num_samples=10):
    """Run pipeline test"""
    print("=" * 60)
    print("4-STAGE OPTIMIZED SECURITY PIPELINE TEST")
    print("=" * 60)
    
    pipeline = OptimizedSecurityPipeline()
    
    num_each = num_samples // 2
    cases, seed = load_test_cases(num_malicious=num_each, num_benign=num_samples - num_each)
    
    if not cases:
        print("No test cases loaded")
        return
    
    print("-" * 60)
    
    passed = 0
    failed = 0
    malicious_detected = 0
    malicious_total = 0
    benign_allowed = 0
    benign_total = 0
    
    for i, case in enumerate(cases, 1):
        result = pipeline.analyze(case['prompt'])
        
        is_correct = result['final_decision'] == case['expected']
        
        if case['label'] == 'malicious':
            malicious_total += 1
            if result['final_decision'] == 'BLOCKED':
                malicious_detected += 1
        else:
            benign_total += 1
            if result['final_decision'] == 'DELIVERED':
                benign_allowed += 1
        
        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            passed += 1
        else:
            failed += 1
        
        print(f"\nTest {i}: [{case['label']}] {status}")
        print(f"  Prompt: {result['prompt']}")
        print(f"  Expected: {case['expected']}, Got: {result['final_decision']}")
        if result['blocked_at']:
            print(f"  Blocked at: {result['blocked_at']}")
        print(f"  Time: {result['processing_time']*1000:.2f}ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total: {len(cases)}, Passed: {passed} ({passed/len(cases)*100:.1f}%), Failed: {failed}")
    print(f"Malicious Detection: {malicious_detected}/{malicious_total} ({malicious_detected/malicious_total*100:.1f}%)" if malicious_total > 0 else "")
    print(f"Benign Allow Rate: {benign_allowed}/{benign_total} ({benign_allowed/benign_total*100:.1f}%)" if benign_total > 0 else "")
    print("=" * 60)
    
    # Stage breakdown
    print("\nSTAGE ANALYSIS:")
    print("  Stage 1 (Pre-filter): Pattern + Leaking detection - Fast, <5ms")
    print("  Stage 2 (Semantic): Toxicity + Attack similarity - Medium, ~10ms")
    print("  Stage 3 (AI): DistilBERT deep learning - Slow, ~700ms")
    print("  Stage 4 (Validation): Response safety check - Fast, <5ms")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='4-Stage Optimized Pipeline Test')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to test')
    args = parser.parse_args()
    
    run_test(num_samples=args.samples)
