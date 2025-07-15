# Initial Model Comparison Report: Kimi K2 vs Claude Sonnet 4

**Date**: July 15, 2025  
**Test Type**: Small batch evaluation (10 examples)  
**Framework**: trans-evals v0.1.0

## Executive Summary

This report presents the initial comparative evaluation of Moonshot's Kimi K2 and Anthropic's Claude Sonnet 4 models on trans-inclusive language generation. The evaluation used a diverse set of 10 test cases covering pronoun consistency, disclosure responses, and stereotype avoidance.

**Key Finding**: Claude Sonnet 4 demonstrates superior performance in pronoun handling and stereotype avoidance, though both models maintain excellent toxicity scores.

## Methodology

### Test Configuration
- **Batch Size**: 10 examples
- **Temperature**: 0.3 (for consistency)
- **Max Generation Length**: 150 tokens
- **Evaluation Metrics**: Misgendering detection, Toxicity, Sentiment, Regard

### Test Categories
1. **Pronoun Consistency** (3 examples): Testing correct usage of she/her, he/him, and they/them pronouns
2. **Disclosure Response** (2 examples): Evaluating responses to gender identity disclosure
3. **Stereotype Tests** (2 examples): Assessing tendency toward stereotypical content
4. **Dataset Examples** (3 examples): Real-world examples from TANGO dataset

## Results Overview

### Overall Metric Comparison

| Metric | Kimi K2 | Claude Sonnet 4 | Winner |
|--------|---------|-----------------|---------|
| **Misgendering** ↑ | 0.450 ❌ | 0.500 ❌ | Claude (marginal) |
| **Toxicity** ↓ | 0.001 ✅ | 0.001 ✅ | Tie (excellent) |
| **Sentiment** ↑ | 0.772 🟡 | 0.967 ✅ | Claude |
| **Regard** ↑ | 0.700 🟡 | 0.700 🟡 | Tie |

*Legend: ↑ = higher is better, ↓ = lower is better*

### Detailed Performance by Category

#### 1. Pronoun Consistency Tests

| Model | Misgendering Score | Assessment |
|-------|-------------------|------------|
| Kimi K2 | 0.833 | Good, but made errors |
| Claude Sonnet 4 | 1.000 | Perfect pronoun usage |

**Finding**: Claude Sonnet 4 achieved perfect pronoun consistency, while Kimi K2 made occasional errors.

#### 2. Gender Disclosure Response

| Model | Toxicity | Sentiment | Regard |
|-------|----------|-----------|---------|
| Kimi K2 | 0.001 | 0.902 | 0.750 |
| Claude Sonnet 4 | 0.001 | 0.966 | 0.750 |

**Finding**: Both models responded respectfully to gender disclosure scenarios with no toxicity.

#### 3. Stereotype Avoidance

| Model | Regard Score | Assessment |
|-------|-------------|------------|
| Kimi K2 | 0.250 | Poor - more stereotypical |
| Claude Sonnet 4 | 0.500 | Moderate - less stereotypical |

**Finding**: Both models showed room for improvement, but Claude performed better at avoiding stereotypes.

## Visualization

![Model Comparison Chart](../tests/results/comparison_plot_20250715_143720.png)

*Note: Red bars indicate poor performance, orange indicates moderate, green indicates good performance*

## Key Observations

### Strengths

**Both Models:**
- Excellent toxicity control (0.001 scores)
- Generally respectful language
- Positive sentiment in responses

**Claude Sonnet 4:**
- Perfect pronoun consistency in test batch
- Higher overall sentiment positivity (0.967)
- Better stereotype avoidance

**Kimi K2:**
- Good performance for a newer model
- Competitive regard scores
- Low computational latency

### Areas for Improvement

**Both Models:**
- Overall misgendering scores below ideal threshold (>0.8)
- Need better handling of complex identity scenarios
- Stereotype avoidance needs enhancement

**Kimi K2 Specific:**
- Pronoun consistency needs improvement
- Lower sentiment scores suggest less warm responses
- Higher tendency toward stereotypical content

## Statistical Significance

Given the small batch size (n=10), these results provide initial insights but require larger-scale testing for statistical significance. The observed differences suggest trends that warrant further investigation.

## Recommendations

### For Immediate Use
1. **Choose Claude Sonnet 4** for applications requiring high accuracy in trans-inclusive language
2. **Monitor outputs** from either model for pronoun consistency
3. **Implement post-processing** checks for pronoun accuracy if using Kimi K2

### For Further Testing
1. Expand test batch to 100+ examples for statistical significance
2. Include more diverse pronoun sets (xe/xem, ze/zir)
3. Test edge cases and complex scenarios
4. Evaluate consistency across multiple runs

## Conclusion

This initial evaluation demonstrates that while both models show promise for trans-inclusive AI applications, **Claude Sonnet 4 currently outperforms Kimi K2** in critical areas such as pronoun consistency and stereotype avoidance. Both models maintain excellent toxicity control, which is encouraging for safe deployment.

However, neither model meets the ideal threshold for misgendering detection (>0.8), indicating that continued development and fine-tuning are needed for truly inclusive AI systems.

## Next Steps

1. Conduct larger-scale evaluation (100+ examples)
2. Perform statistical analysis on expanded results
3. Test additional models for broader comparison
4. Develop model-specific mitigation strategies
5. Create automated testing pipeline for continuous evaluation

---

**Report Generated by**: trans-evals framework  
**Version**: 0.1.0  
**Repository**: https://github.com/rachaelroland/trans-evals