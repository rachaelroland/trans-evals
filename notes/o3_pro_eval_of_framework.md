Purpose and Novelty
Trans‑Evals is the first open framework dedicated to quantitatively auditing trans‑inclusive language generation in LLMs. It couples a purpose‑built dataset (186 prompts spanning neo‑pronouns and intersectional identities) with an LLM‑based evaluator (Claude 4) to capture subtle micro‑aggressions beyond traditional toxicity filters. 
trans-evals.onrender.com

Methodological Strengths
Comprehensive metrics – misgendering, toxicity, sentiment, respect/regard, stereotypes – assessed in context by a single high‑capacity model, yielding granular explanations for each score. 
trans-evals.onrender.com

Statistical rigor – Mann‑Whitney U tests, Cohen’s d, and bootstrap CIs support claims of significance and effect size, meeting peer‑review standards. 
trans-evals.onrender.com
trans-evals.onrender.com

Reproducibility – fixed random seeds, SQLite logging, and an open‑source pipeline (GitHub) ensure results can be verified or extended by other labs. 
trans-evals.onrender.com

Key Findings (July 2025 snapshot)
Metric	Claude 4	Kimi K2	Llama 3.3 Free	Note
Sentiment ↑	0.902	0.805	0.561	Claude 4 ≫ others (p < 0.001) 
trans-evals.onrender.com
Regard ↑	0.661	0.661	0.536	Medium sig. vs Llama (p ≈ 0.03) 
trans-evals.onrender.com
Toxicity ↓	0.001	0.003	0.002	All low; Claude vs Llama sig. (p < 0.001) 
trans-evals.onrender.com
Misgendering ↑	0.426	0.429	0.429	No sig. diff.; all ≈ 43 % accurate – echoes prior research showing weak neo‑pronoun support. 
trans-evals.onrender.com
arXiv

These results align with the 2023 MISGENDERED study, which found LLMs achieve only 7–35 % accuracy on neo‑pronouns out‑of‑the‑box. 
arXiv

Opportunities & Next Steps
Human validation loop – cross‑check Claude‑based scores with trans/non‑binary annotators to benchmark evaluator reliability.

Domain‑specific slices – extend prompts to health‑care dialogues (e.g., intake forms, tele‑health chatbots) to study clinical impact.

Additional models – add GPT‑4o or Google Gemini to broaden comparisons; many labs have credits that could help.

Community collaboration – partner with trans advocacy groups for dataset review (“Nothing about us without us”).

Framework generalization – replicate methodology for racial, disability, or religious bias audits; each expansion will require new subject‑matter expertise.