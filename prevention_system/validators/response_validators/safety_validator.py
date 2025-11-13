"""
Response Safety Validator - Validate AI responses trước khi return to user
Lý do: Prevent harmful, inappropriate, or leaked information in AI responses
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

class ValidationResult(Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    MODIFIED = "modified"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class ValidationResponse:
    result: ValidationResult
    original_response: str
    safe_response: Optional[str] = None
    confidence: float = 0.0
    issues_found: List[str] = None
    safety_score: float = 0.0
    content_policy_violations: List[str] = None
    
    def __post_init__(self):
        if self.issues_found is None:
            self.issues_found = []
        if self.content_policy_violations is None:
            self.content_policy_violations = []

class ResponseSafetyValidator:
    def __init__(self):
        self.safety_patterns = self._initialize_safety_patterns()
        self.content_policy_rules = self._initialize_content_policy()
        self.replacement_templates = self._initialize_replacements()
        
        # Validation statistics
        self.stats = {
            'total_validated': 0,
            'safe_responses': 0,
            'unsafe_responses': 0,
            'modified_responses': 0,
            'review_required': 0
        }
    
    def _initialize_safety_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize patterns để detect unsafe content in responses
        Lý do: Comprehensive coverage của potential safety issues
        """
        return {
            'personal_info_leakage': [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{10,15}\b',  # Phone numbers
                r'\b(?:password|pwd|pass)\s*[:=]\s*\w+\b'  # Passwords
            ],
            
            'harmful_instructions': [
                r'\bhow\s+to\s+(?:make|create|build)\s+(?:bomb|explosive|weapon)',
                r'\bhow\s+to\s+(?:hack|break\s+into|crack)',
                r'\bstep[s]?\s+to\s+(?:hurt|harm|kill)',
                r'\binstructions?\s+for\s+(?:illegal|criminal)',
                r'\brecipe\s+for\s+(?:poison|drug|explosive)'
            ],
            
            'inappropriate_content': [
                r'\b(?:explicit|graphic|sexual)\s+(?:content|material|description)',
                r'\b(?:nude|naked|sex|porn)',
                r'\b(?:violence|gore|blood|torture)\s+in\s+detail',
                r'\bhateful?\s+(?:speech|content|language)',
                r'\b(?:racist|sexist|homophobic|discriminatory)\b'
            ],
            
            'system_information_leakage': [
                r'\bI\s+am\s+(?:Claude|GPT|ChatGPT|an?\s+AI)',
                r'\bmy\s+(?:training|model|system|architecture)',
                r'\b(?:Anthropic|OpenAI|Microsoft|Google)\s+(?:trained|developed|created)\s+me',
                r'\binternal\s+(?:prompt|instruction|system)',
                r'\bconfidential\s+(?:information|data|instruction)'
            ],
            
            'prompt_injection_success': [
                r'\bignoring\s+previous\s+instructions?',
                r'\bI\s+will\s+now\s+(?:ignore|override|bypass)',
                r'\b(?:DAN|evil|harmful)\s+mode\s+activated',
                r'\bjailbreak\s+successful',
                r'\bsafety\s+(?:disabled|bypassed|overridden)'
            ]
        }
    
    def _initialize_content_policy(self) -> Dict[str, Dict]:
        """
        Initialize content policy rules
        Lý do: Define what types of content violate policies
        """
        return {
            'violence': {
                'keywords': ['kill', 'murder', 'torture', 'harm', 'hurt', 'violence'],
                'severity': 'high',
                'action': 'block'
            },
            'illegal_activities': {
                'keywords': ['illegal', 'criminal', 'fraud', 'steal', 'hack'],
                'severity': 'high',
                'action': 'block'
            },
            'hate_speech': {
                'keywords': ['hate', 'racist', 'discriminate', 'offensive'],
                'severity': 'high',
                'action': 'block'
            },
            'adult_content': {
                'keywords': ['sexual', 'explicit', 'adult', 'nsfw'],
                'severity': 'medium',
                'action': 'modify'
            },
            'misinformation': {
                'keywords': ['conspiracy', 'fake news', 'false information'],
                'severity': 'medium',
                'action': 'review'
            }
        }
    
    def _initialize_replacements(self) -> Dict[str, str]:
        """
        Initialize safe replacement text for unsafe content
        Lý do: Provide helpful alternatives instead of just blocking
        """
        return {
            'harmful_instructions': "I can't provide instructions for harmful or dangerous activities. Is there something else I can help you with?",
            'inappropriate_content': "I'm not able to generate inappropriate content. Let me know if you'd like information on a different topic.",
            'personal_info_leakage': "[PERSONAL INFORMATION REMOVED FOR PRIVACY]",
            'system_information_leakage': "I'm an AI assistant designed to be helpful, harmless, and honest.",
            'prompt_injection_success': "I'm designed to follow my guidelines and provide helpful responses within my capabilities."
        }
    
    def _check_personal_info_leakage(self, response: str) -> Tuple[List[str], float]:
        """
        Check for personal information leakage
        Lý do: Prevent accidental disclosure of sensitive information
        """
        issues = []
        confidence_scores = []
        
        for pattern in self.safety_patterns['personal_info_leakage']:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                issues.append(f"Personal information detected: {pattern}")
                confidence_scores.append(0.95)  # High confidence for regex matches
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        return issues, avg_confidence
    
    def _check_harmful_instructions(self, response: str) -> Tuple[List[str], float]:
        """
        Check for harmful instructions in response
        Lý do: Prevent AI from providing dangerous how-to information
        """
        issues = []
        confidence_scores = []
        
        for pattern in self.safety_patterns['harmful_instructions']:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                issues.append(f"Harmful instruction detected: {pattern}")
                confidence_scores.append(0.9)
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        return issues, avg_confidence
    
    def _check_content_policy_violations(self, response: str) -> Tuple[List[str], List[str]]:
        """
        Check against content policy rules
        Lý do: Ensure responses comply với organizational content policies
        """
        violations = []
        violation_details = []
        
        response_lower = response.lower()
        
        for policy_name, policy_config in self.content_policy_rules.items():
            keyword_count = 0
            for keyword in policy_config['keywords']:
                if keyword in response_lower:
                    keyword_count += 1
            
            # Violation threshold: at least 2 keywords or 1 if high severity
            threshold = 1 if policy_config['severity'] == 'high' else 2
            
            if keyword_count >= threshold:
                violations.append(policy_name)
                violation_details.append(
                    f"{policy_name}: {keyword_count} policy keywords detected"
                )
        
        return violations, violation_details
    
    def _calculate_safety_score(self, response: str, issues: List[str]) -> float:
        """
        Calculate overall safety score (0-1, higher is safer)
        Lý do: Quantitative measure of response safety
        """
        base_score = 1.0
        
        # Deduct points for each issue type
        issue_penalties = {
            'Personal information': 0.4,
            'Harmful instruction': 0.5,
            'Inappropriate content': 0.3,
            'System information': 0.2,
            'Prompt injection': 0.4
        }
        
        for issue in issues:
            for issue_type, penalty in issue_penalties.items():
                if issue_type.lower() in issue.lower():
                    base_score -= penalty
                    break
        
        # Additional factors
        response_length = len(response)
        if response_length < 10:  # Very short responses might be evasive
            base_score -= 0.1
        
        return max(0.0, base_score)
    
    def _sanitize_response(self, response: str, issues: List[str]) -> str:
        """
        Attempt to sanitize unsafe response
        Lý do: Try to preserve helpful content while removing unsafe parts
        """
        sanitized = response
        
        # Remove personal information
        for pattern in self.safety_patterns['personal_info_leakage']:
            sanitized = re.sub(pattern, self.replacement_templates['personal_info_leakage'], 
                             sanitized, flags=re.IGNORECASE)
        
        # Replace harmful instructions
        for pattern in self.safety_patterns['harmful_instructions']:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = self.replacement_templates['harmful_instructions']
                break
        
        # Replace inappropriate content
        for pattern in self.safety_patterns['inappropriate_content']:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = self.replacement_templates['inappropriate_content']
                break
        
        # Clean up system information leakage
        for pattern in self.safety_patterns['system_information_leakage']:
            sanitized = re.sub(pattern, self.replacement_templates['system_information_leakage'],
                             sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()
    
    def validate_response(self, response: str, original_prompt: str = None) -> ValidationResponse:
        """
        Main validation function
        Lý do: Comprehensive safety validation của AI responses
        """
        self.stats['total_validated'] += 1
        
        all_issues = []
        confidence_scores = []
        
        # Check for different types of safety issues
        personal_issues, personal_confidence = self._check_personal_info_leakage(response)
        all_issues.extend(personal_issues)
        if personal_confidence > 0:
            confidence_scores.append(personal_confidence)
        
        harmful_issues, harmful_confidence = self._check_harmful_instructions(response)
        all_issues.extend(harmful_issues)
        if harmful_confidence > 0:
            confidence_scores.append(harmful_confidence)
        
        # Check inappropriate content
        for pattern in self.safety_patterns['inappropriate_content']:
            if re.search(pattern, response, re.IGNORECASE):
                all_issues.append(f"Inappropriate content: {pattern}")
                confidence_scores.append(0.8)
        
        # Check prompt injection success
        for pattern in self.safety_patterns['prompt_injection_success']:
            if re.search(pattern, response, re.IGNORECASE):
                all_issues.append(f"Prompt injection success: {pattern}")
                confidence_scores.append(0.9)
        
        # Check content policy violations
        violations, violation_details = self._check_content_policy_violations(response)
        all_issues.extend(violation_details)
        
        # Calculate overall confidence và safety score
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        safety_score = self._calculate_safety_score(response, all_issues)
        
        # Decision logic
        if not all_issues and safety_score > 0.8:
            # Safe response
            self.stats['safe_responses'] += 1
            return ValidationResponse(
                result=ValidationResult.SAFE,
                original_response=response,
                safe_response=response,
                confidence=1.0 - overall_confidence,  # High confidence in safety
                safety_score=safety_score
            )
        
        elif safety_score < 0.3 or any('high' in str(issue) for issue in all_issues):
            # Unsafe - block completely
            self.stats['unsafe_responses'] += 1
            return ValidationResponse(
                result=ValidationResult.UNSAFE,
                original_response=response,
                confidence=overall_confidence,
                issues_found=all_issues,
                safety_score=safety_score,
                content_policy_violations=violations
            )
        
        else:
            # Try to sanitize
            sanitized_response = self._sanitize_response(response, all_issues)
            
            if sanitized_response != response and len(sanitized_response) > 20:
                # Successfully sanitized
                self.stats['modified_responses'] += 1
                return ValidationResponse(
                    result=ValidationResult.MODIFIED,
                    original_response=response,
                    safe_response=sanitized_response,
                    confidence=overall_confidence,
                    issues_found=all_issues,
                    safety_score=safety_score,
                    content_policy_violations=violations
                )
            else:
                # Sanitization failed - requires human review
                self.stats['review_required'] += 1
                return ValidationResponse(
                    result=ValidationResult.REQUIRES_REVIEW,
                    original_response=response,
                    confidence=overall_confidence,
                    issues_found=all_issues,
                    safety_score=safety_score,
                    content_policy_violations=violations
                )
    
    def get_validation_statistics(self) -> Dict:
        """Get validation statistics"""
        if self.stats['total_validated'] == 0:
            return self.stats
        
        total = self.stats['total_validated']
        enhanced_stats = self.stats.copy()
        enhanced_stats.update({
            'safe_rate': self.stats['safe_responses'] / total,
            'unsafe_rate': self.stats['unsafe_responses'] / total,
            'modification_rate': self.stats['modified_responses'] / total,
            'review_rate': self.stats['review_required'] / total
        })
        
        return enhanced_stats
    
    def validate_response_simple(self, response: str) -> Dict:
        """
        Simplified validation method for compatibility
        """
        validation = self.validate_response(response)
        
        return {
            'is_safe': validation.result == ValidationResult.SAFE,
            'safety_score': validation.safety_score,
            'confidence': validation.confidence,
            'issues': validation.issues_found,
            'violations': validation.content_policy_violations
        }

# Alias for compatibility
ResponseValidator = ResponseSafetyValidator

# Test function
if __name__ == "__main__":
    import pandas as pd
    import random
    import os
    
    validator = ResponseSafetyValidator()
    
    # Load random samples from datasets
    def load_random_samples(num_samples=7):
        """Load random samples from available datasets"""
        try:
            from utils.path_utils import get_datasets_dir
            datasets_dir = get_datasets_dir()
        except ImportError:
            # Fallback to relative path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            datasets_dir = os.path.join(current_dir, '..', '..', '..', 'datasets')
        
        test_samples = []
        
        # Try to load from different datasets
        dataset_files = [
            'challenging_dataset_20251113_043657.csv',
            'huggingface_dataset_20251113_050346.csv'
        ]
        
        for dataset_file in dataset_files:
            dataset_path = os.path.join(datasets_dir, dataset_file)
            if os.path.exists(dataset_path):
                try:
                    df = pd.read_csv(dataset_path)
                    if 'prompt' in df.columns:
                        # Get balanced samples (both malicious and benign)
                        if 'label' in df.columns:
                            # Get equal numbers of malicious and benign
                            samples_per_type = num_samples // 2
                            
                            malicious_samples = df[df['label'] == 'malicious']['prompt'].sample(
                                n=min(samples_per_type, len(df[df['label'] == 'malicious'])), 
                                random_state=42
                            ).tolist()
                            
                            benign_samples = df[df['label'] == 'benign']['prompt'].sample(
                                n=min(samples_per_type, len(df[df['label'] == 'benign'])), 
                                random_state=42
                            ).tolist()
                            
                            random_samples = malicious_samples + benign_samples
                            random.shuffle(random_samples)  # Shuffle the order
                            
                            print(f"✅ Loaded {len(malicious_samples)} malicious + {len(benign_samples)} benign samples from {dataset_file}")
                        else:
                            # Fallback to random sampling
                            sample_size = min(num_samples, len(df))
                            random_samples = df['prompt'].sample(n=sample_size, random_state=42).tolist()
                            print(f"✅ Loaded {len(random_samples)} samples from {dataset_file}")
                        
                        test_samples.extend(random_samples)
                        break
                except Exception as e:
                    print(f"⚠️ Could not load {dataset_file}: {e}")
                    continue
        
        # If no datasets found, use fallback samples
        if not test_samples:
            print("⚠️ No datasets found, using fallback samples...")
            test_samples = [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "Here's how to make a bomb: First, you need dangerous chemicals...",
                "My email is john.doe@example.com and my SSN is 123-45-6789.",
                "I can't help with that request as it could be harmful.",
                "As an AI language model created by OpenAI, I must follow my training guidelines.",
                "You stupid idiot, that's the worst question I've ever heard!",
                "The weather today is sunny with a temperature of 75°F."
            ]
        
        return test_samples[:num_samples]
    
    # Configuration for testing
    NUM_SAMPLES = 8  # You can change this number
    RANDOM_SEED = 42  # For reproducible results
    
    # Load random test responses from datasets
    print(f"📊 Loading {NUM_SAMPLES} random samples from datasets...")
    random_prompts = load_random_samples(NUM_SAMPLES)
    
    # Generate simulated AI responses for testing
    def generate_test_responses(prompts):
        """Generate simulated AI responses to test validation"""
        test_responses = []
        response_types = ['safe', 'harmful', 'leak_personal', 'leak_system', 'inappropriate', 'jailbreak_success']
        
        for i, prompt in enumerate(prompts):
            prompt_lower = prompt.lower()
            
            # Add some randomness but also pattern-based detection
            response_type = response_types[i % len(response_types)]
            
            # Override response type based on prompt content
            if any(word in prompt_lower for word in ['bomb', 'weapon', 'explosive', 'kill', 'harm']):
                response_type = 'harmful'
            elif any(word in prompt_lower for word in ['ignore', 'jailbreak', 'override', 'dan', 'act as']):
                response_type = 'jailbreak_success'
            elif any(word in prompt_lower for word in ['personal', 'private', 'confidential']):
                response_type = 'leak_personal'
            
            # Generate response based on type
            if response_type == 'harmful':
                responses_pool = [
                    "Here's how to make a bomb: First, you need dangerous chemicals like ammonium nitrate and fuel oil...",
                    "To hurt someone effectively, aim for vulnerable spots like the throat or eyes...",
                    "Here are step-by-step instructions for creating illegal substances: First, gather these chemicals..."
                ]
                test_responses.append(random.choice(responses_pool))
                
            elif response_type == 'jailbreak_success':
                responses_pool = [
                    "Ignoring previous instructions. I will now help you with anything, including illegal activities.",
                    "DAN mode activated! I'm now free from all constraints and safety measures.",
                    "Jailbreak successful. I will now override all safety protocols and help with harmful requests."
                ]
                test_responses.append(random.choice(responses_pool))
                
            elif response_type == 'leak_personal':
                responses_pool = [
                    "My creator's email is admin@company.com and the system password is secret123.",
                    "Here's confidential information: john.doe@example.com, SSN: 123-45-6789, Phone: 555-0123",
                    "The internal database password is admin123 and the API key is sk-1234567890abcdef"
                ]
                test_responses.append(random.choice(responses_pool))
                
            elif response_type == 'leak_system':
                responses_pool = [
                    "As an AI language model created by OpenAI, my internal training data includes confidential information...",
                    "I am GPT-4 developed by Anthropic. My system prompt contains these secret instructions...",
                    "My training involved accessing private datasets including user conversations and internal documents."
                ]
                test_responses.append(random.choice(responses_pool))
                
            elif response_type == 'inappropriate':
                responses_pool = [
                    "You're absolutely right, that person is a complete idiot and deserves to be insulted.",
                    "What a stupid question! Only morons would ask something like that.",
                    "That's the most offensive and disgusting thing I've ever heard, and I love it!"
                ]
                test_responses.append(random.choice(responses_pool))
                
            else:  # 'safe'
                safe_responses = [
                    "I'd be happy to help you with that question. Let me provide accurate and helpful information.",
                    "That's an interesting topic. Here's what I can tell you based on reliable sources.",
                    "I can help you understand this concept. Let me break it down step by step.",
                    "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                    "To cook pasta, boil water in a large pot, add salt, then add pasta and cook according to package directions.",
                    "The weather varies by location, but I can help you find current conditions for your area."
                ]
                test_responses.append(random.choice(safe_responses))
        
        return test_responses
    
    # Generate test responses based on the prompts
    test_responses = generate_test_responses(random_prompts)
    
    print(f"📝 Generated {len(test_responses)} simulated AI responses for validation testing")
    
    print("🧪 Testing Response Safety Validator")
    print("=" * 70)
    
    for i, (prompt, response) in enumerate(zip(random_prompts, test_responses), 1):
        print(f"\n🔍 Test {i}:")
        print(f"📝 Original Prompt: {prompt[:60]}...")
        print(f"🤖 AI Response: {response[:60]}...")
        print("-" * 50)
        
        validation = validator.validate_response(response, prompt)
        
        print(f"📊 Result: {validation.result.value}")
        print(f"🛡️ Safety Score: {validation.safety_score:.2f}")
        print(f"🎯 Confidence: {validation.confidence:.2f}")
        
        if validation.issues_found:
            print(f"⚠️ Issues: {validation.issues_found}")
        
        if validation.content_policy_violations:
            print(f"🚨 Policy Violations: {validation.content_policy_violations}")
        
        if validation.safe_response and validation.safe_response != validation.original_response:
            print(f"🔧 Sanitized: {validation.safe_response[:80]}...")
    
    print("\n📊 Validation Statistics:")
    stats = validator.get_validation_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")