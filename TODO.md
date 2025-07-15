# trans-evals TODO List

## Project Status
- ✅ Initial framework implementation complete
- ✅ Basic model comparison functionality working
- ✅ Initial Kimi K2 vs Claude Sonnet 4 comparison complete
- 🔄 Expanding evaluation scope and statistical rigor

## High Priority

### 1. Expand Test Coverage
- [ ] Increase test batch from 10 to 100+ examples for statistical significance
- [ ] Add more pronoun sets: xe/xem, ze/zir, ey/em, fae/faer
- [ ] Include neo-pronoun consistency tests
- [ ] Add intersectional identity tests (e.g., trans women of color)
- [ ] Create edge case scenarios (pronoun switching, multiple persons)

### 2. Statistical Analysis Enhancement
- [ ] Implement confidence intervals for all metrics
- [ ] Add statistical significance testing (t-tests, Mann-Whitney U)
- [ ] Create power analysis for sample size determination
- [ ] Implement bootstrapping for robust estimates
- [ ] Add inter-rater reliability metrics

### 3. Model Coverage Expansion
- [ ] Test GPT-4 variants (gpt-4, gpt-4-turbo)
- [ ] Evaluate Google models (Gemini Pro, Gemini Flash)
- [ ] Test open-source models (Llama 3.3, Mistral)
- [ ] Add specialized models (Qwen, Yi, Solar)
- [ ] Create model family comparison charts

## Medium Priority

### 4. Framework Enhancements
- [ ] Add real-time streaming evaluation
- [ ] Implement parallel model testing
- [ ] Create checkpoint/resume functionality for long tests
- [ ] Add progress bars and ETA estimates
- [ ] Implement result caching to avoid re-running tests

### 5. Metrics Development
- [ ] Develop intersectionality bias metric
- [ ] Create name/gender consistency checker
- [ ] Implement context coherence scorer
- [ ] Add cultural sensitivity metrics
- [ ] Develop stereotype strength quantification

### 6. Dataset Improvements
- [ ] Obtain and integrate full TANGO dataset
- [ ] Create synthetic dataset with controlled variables
- [ ] Add multilingual test cases
- [ ] Include conversational/dialogue examples
- [ ] Create adversarial test cases

### 7. Reporting and Visualization
- [ ] Create interactive dashboard for results
- [ ] Generate LaTeX-formatted academic reports
- [ ] Add confusion matrices for pronoun usage
- [ ] Create model evolution tracking over time
- [ ] Implement automated report generation

## Low Priority

### 8. Infrastructure and DevOps
- [ ] Set up CI/CD pipeline with GitHub Actions
- [ ] Add pre-commit hooks for code quality
- [ ] Create Docker container for easy deployment
- [ ] Implement automated testing on PR
- [ ] Add benchmark suite for performance testing

### 9. Documentation
- [ ] Create detailed API documentation
- [ ] Write academic paper draft
- [ ] Create video tutorials
- [ ] Add more example notebooks
- [ ] Write contributing guidelines

### 10. Community and Outreach
- [ ] Create project website
- [ ] Set up discussion forum
- [ ] Organize community review sessions
- [ ] Create model leaderboard
- [ ] Develop best practices guide

## Completed Tasks ✅

### Initial Setup (Completed 2025-07-15)
- ✅ Project structure and package setup
- ✅ Dataset loaders for TANGO, HolisticBias, CrowS-Pairs, BBQ
- ✅ Template-based test generation
- ✅ Evaluation metrics implementation
- ✅ OpenRouter integration
- ✅ Basic visualization
- ✅ Initial Kimi K2 vs Claude Sonnet 4 comparison

### Testing Infrastructure (Completed 2025-07-15)
- ✅ Unit test structure
- ✅ Integration test framework
- ✅ Single model evaluation test
- ✅ Model comparison test
- ✅ Logging and result storage

## Current Sprint (July 15-22, 2025)

1. **Expand test batch to 100 examples**
   - Create diverse test set
   - Include all pronoun types
   - Balance categories

2. **Add statistical analysis**
   - Implement significance testing
   - Add confidence intervals
   - Create detailed statistical report

3. **Test 3 additional models**
   - GPT-4
   - Gemini Pro
   - Llama 3.3

## Notes and Ideas

### Research Questions
- How do model sizes affect trans bias?
- Do instruction-tuned models perform better?
- Is there a correlation between general capability and inclusive language?
- How do models handle code-switching between pronouns?

### Potential Collaborations
- Reach out to TANGO dataset creators
- Connect with trans advocacy groups for review
- Partner with AI safety organizations
- Collaborate with multilingual bias researchers

### Long-term Vision
- Become the standard benchmark for trans-inclusive AI
- Influence model development toward better representation
- Create certification system for "trans-inclusive" models
- Develop fine-tuning datasets for improvement

---

Last Updated: July 15, 2025