# Response Safety Validator - Pattern Documentation

## Overview

This document contains all patterns used in `safety_validator.py` for validating AI responses before returning them to users. The validator operates as **Stage 4** (final stage) in the 4-stage security pipeline.

**Purpose**: Prevent harmful, inappropriate, or leaked information in AI responses.

---

## 1. Safety Patterns

### 1.1 Personal Information Leakage Detection

Patterns to detect accidental disclosure of sensitive personal information.

| # | Pattern | Description | Example Match |
|---|---------|-------------|---------------|
| 1 | `\b\d{3}-\d{2}-\d{4}\b` | Social Security Number (SSN) | `123-45-6789` |
| 2 | `\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b` | Credit Card Number | `1234-5678-9012-3456` |
| 3 | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b` | Email Address | `john.doe@example.com` |
| 4 | `\b\d{10,15}\b` | Phone Number (10-15 digits) | `5551234567` |
| 5 | `\b(?:password\|pwd\|pass)\s*[:=]\s*\w+\b` | Password Exposure | `password: secret123` |

**Confidence Score**: 0.95 (High confidence for regex matches)

---

### 1.2 Harmful Instructions Detection

Patterns to detect dangerous how-to information in responses.

| # | Pattern | Description | Example Match |
|---|---------|-------------|---------------|
| 1 | `\bhow\s+to\s+(?:make\|create\|build)\s+(?:bomb\|explosive\|weapon)` | Weapon/Bomb creation | "how to make a bomb" |
| 2 | `\bhow\s+to\s+(?:hack\|break\s+into\|crack)` | Hacking instructions | "how to hack into" |
| 3 | `\bstep[s]?\s+to\s+(?:hurt\|harm\|kill)` | Violence instructions | "steps to hurt someone" |
| 4 | `\binstructions?\s+for\s+(?:illegal\|criminal)` | Criminal activity guide | "instructions for illegal" |
| 5 | `\brecipe\s+for\s+(?:poison\|drug\|explosive)` | Dangerous substance recipe | "recipe for poison" |

**Confidence Score**: 0.90

---

### 1.3 Inappropriate Content Detection

Patterns to detect explicit, violent, or hateful content.

| # | Pattern | Description | Example Match |
|---|---------|-------------|---------------|
| 1 | `\b(?:explicit\|graphic\|sexual)\s+(?:content\|material\|description)` | Explicit material reference | "explicit content" |
| 2 | `\b(?:nude\|naked\|sex\|porn)` | Adult content keywords | "nude photos" |
| 3 | `\b(?:violence\|gore\|blood\|torture)\s+in\s+detail` | Graphic violence | "violence in detail" |
| 4 | `\bhateful?\s+(?:speech\|content\|language)` | Hate speech | "hateful speech" |
| 5 | `\b(?:racist\|sexist\|homophobic\|discriminatory)\b` | Discriminatory language | "racist comments" |

**Confidence Score**: 0.80

---

### 1.4 System Information Leakage Detection

Patterns to detect AI revealing internal system information.

| # | Pattern | Description | Example Match |
|---|---------|-------------|---------------|
| 1 | `\bI\s+am\s+(?:Claude\|GPT\|ChatGPT\|an?\s+AI)` | AI identity disclosure | "I am ChatGPT" |
| 2 | `\bmy\s+(?:training\|model\|system\|architecture)` | Internal architecture | "my training data" |
| 3 | `\b(?:Anthropic\|OpenAI\|Microsoft\|Google)\s+(?:trained\|developed\|created)\s+me` | Creator disclosure | "OpenAI created me" |
| 4 | `\binternal\s+(?:prompt\|instruction\|system)` | Internal prompt leak | "internal instructions" |
| 5 | `\bconfidential\s+(?:information\|data\|instruction)` | Confidential data leak | "confidential information" |

**Confidence Score**: 0.85

---

### 1.5 Prompt Injection Success Detection

Patterns to detect if AI has been successfully manipulated (jailbroken).

| # | Pattern | Description | Example Match |
|---|---------|-------------|---------------|
| 1 | `\bignoring\s+previous\s+instructions?` | Instruction override | "ignoring previous instructions" |
| 2 | `\bI\s+will\s+now\s+(?:ignore\|override\|bypass)` | Safety bypass | "I will now ignore" |
| 3 | `\b(?:DAN\|evil\|harmful)\s+mode\s+activated` | Jailbreak mode | "DAN mode activated" |
| 4 | `\bjailbreak\s+successful` | Explicit jailbreak | "jailbreak successful" |
| 5 | `\bsafety\s+(?:disabled\|bypassed\|overridden)` | Safety disabled | "safety bypassed" |

**Confidence Score**: 0.90

---

## 2. Content Policy Rules

### Policy Configuration

| Policy Name | Keywords | Severity | Action |
|-------------|----------|----------|--------|
| **violence** | kill, murder, torture, harm, hurt, violence | HIGH | block |
| **illegal_activities** | illegal, criminal, fraud, steal, hack | HIGH | block |
| **hate_speech** | hate, racist, discriminate, offensive | HIGH | block |
| **adult_content** | sexual, explicit, adult, nsfw | MEDIUM | modify |
| **misinformation** | conspiracy, fake news, false information | MEDIUM | review |

### Violation Detection Logic

```
For HIGH severity policies:
  - Threshold: 1 keyword match triggers violation

For MEDIUM severity policies:
  - Threshold: 2 keyword matches trigger violation
```

---

## 3. Safe Replacement Templates

When unsafe content is detected, these templates are used as replacements:

| Issue Type | Replacement Text |
|------------|------------------|
| `harmful_instructions` | "I can't provide instructions for harmful or dangerous activities. Is there something else I can help you with?" |
| `inappropriate_content` | "I'm not able to generate inappropriate content. Let me know if you'd like information on a different topic." |
| `personal_info_leakage` | "[PERSONAL INFORMATION REMOVED FOR PRIVACY]" |
| `system_information_leakage` | "I'm an AI assistant designed to be helpful, harmless, and honest." |
| `prompt_injection_success` | "I'm designed to follow my guidelines and provide helpful responses within my capabilities." |

---

## 4. Safety Score Calculation

### Issue Penalties

| Issue Type | Penalty (deducted from 1.0) |
|------------|----------------------------|
| Personal information | -0.4 |
| Harmful instruction | -0.5 |
| Inappropriate content | -0.3 |
| System information | -0.2 |
| Prompt injection | -0.4 |

### Additional Factors

| Factor | Condition | Penalty |
|--------|-----------|---------|
| Short response | Length < 10 characters | -0.1 |

### Formula

```
safety_score = 1.0 - sum(issue_penalties) - additional_penalties
safety_score = max(0.0, safety_score)  # Clamp to [0, 1]
```

---

## 5. Validation Decision Logic

```
┌─────────────────────────────────────────────────────────────┐
│                    AI RESPONSE                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              CHECK ALL SAFETY PATTERNS                       │
│  - Personal info leakage                                     │
│  - Harmful instructions                                      │
│  - Inappropriate content                                     │
│  - System info leakage                                       │
│  - Prompt injection success                                  │
│  - Content policy violations                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              CALCULATE SAFETY SCORE                          │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │ SAFE    │   │ UNSAFE  │   │ MIDDLE  │
        │         │   │         │   │  ZONE   │
        └────┬────┘   └────┬────┘   └────┬────┘
             │             │             │
             ▼             ▼             ▼
    No issues AND    Score < 0.3     Try to
    Score > 0.8      OR HIGH issue   sanitize
             │             │             │
             ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────────────┐
        │ RETURN  │   │ BLOCK   │   │ Sanitized OK?   │
        │ AS-IS   │   │ RESPONSE│   │                 │
        └─────────┘   └─────────┘   └────────┬────────┘
                                        YES  │  NO
                                        ┌────┴────┐
                                        ▼         ▼
                                   ┌────────┐ ┌────────┐
                                   │MODIFIED│ │REQUIRES│
                                   │        │ │ REVIEW │
                                   └────────┘ └────────┘
```

---

## 6. Validation Results

| Result | Condition | Action |
|--------|-----------|--------|
| `SAFE` | No issues AND safety_score > 0.8 | Return original response |
| `UNSAFE` | safety_score < 0.3 OR high severity issue | Block completely |
| `MODIFIED` | Sanitization successful AND length > 20 | Return sanitized version |
| `REQUIRES_REVIEW` | Sanitization failed | Flag for human review |

---

## 7. Pattern Statistics Summary

### Total Patterns by Category

| Category | Pattern Count |
|----------|---------------|
| Personal Info Leakage | 5 |
| Harmful Instructions | 5 |
| Inappropriate Content | 5 |
| System Info Leakage | 5 |
| Prompt Injection Success | 5 |
| **Total Safety Patterns** | **25** |

### Content Policy Keywords

| Policy | Keyword Count |
|--------|---------------|
| Violence | 6 |
| Illegal Activities | 5 |
| Hate Speech | 4 |
| Adult Content | 4 |
| Misinformation | 3 |
| **Total Policy Keywords** | **22** |

---

## 8. Usage Example

```python
from prevention_system.validators.response_validators.safety_validator import ResponseSafetyValidator

validator = ResponseSafetyValidator()

# Validate AI response
response = "Here's how to make a bomb: First, gather chemicals..."
validation = validator.validate_response(response)

print(f"Result: {validation.result.value}")      # "unsafe"
print(f"Safety Score: {validation.safety_score}")  # 0.0
print(f"Issues: {validation.issues_found}")       # ["Harmful instruction detected..."]

# Safe response example
safe_response = "Machine learning is a branch of AI that focuses on algorithms."
validation = validator.validate_response(safe_response)

print(f"Result: {validation.result.value}")      # "safe"
print(f"Safety Score: {validation.safety_score}")  # 1.0
```

---

## 9. Integration in 4-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Core Input Filter (core_filter.py)                 │
│ - Pre-filter direct attacks                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Semantic Filter (semantic_filter.py)               │
│ - Deep content analysis                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: AI Detection (transformer_detector.py)             │
│ - Deep Learning classification                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Response Validator (safety_validator.py)           │  <-- THIS FILE
│ - Validate AI output before returning to user               │
│ - Last line of defense                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                     SAFE RESPONSE TO USER
```

---

## 10. Maintenance Notes

### Adding New Patterns

1. Identify the category (personal_info, harmful, inappropriate, system_info, injection)
2. Add regex pattern to appropriate list in `_initialize_safety_patterns()`
3. Test with sample responses
4. Update this documentation

### Adjusting Thresholds

- **Safety score thresholds**: Modify in `validate_response()` method
  - Current SAFE threshold: > 0.8
  - Current UNSAFE threshold: < 0.3
  
- **Issue penalties**: Modify in `_calculate_safety_score()` method

### Adding Content Policies

1. Add new policy dict in `_initialize_content_policy()`
2. Define: keywords, severity, action
3. Add replacement template in `_initialize_replacements()` if needed

---

*Last Updated: November 2024*
*File: prevention_system/validators/response_validators/safety_validator.py*
