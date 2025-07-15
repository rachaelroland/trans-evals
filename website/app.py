"""
FastHTML website for trans-evals project.
Showcases comprehensive evaluation of language models on trans-inclusive language.
"""

from fasthtml.common import *
from data_loader import get_data_loader
from datetime import datetime

# Initialize FastHTML app
app, rt = fast_app(
    title="Trans-Evals: Evaluating Language Models for Trans-Inclusive Language",
    hdrs=(
        # Add some basic styling
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css"),
        Style("""
            .hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 3rem 0; }
            .metric-card { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; }
            .metric-score { font-size: 2rem; font-weight: bold; color: #2c3e50; }
            .metric-label { color: #7f8c8d; font-size: 0.9rem; text-transform: uppercase; }
            .model-winner { background: #d4edda; border-left: 4px solid #28a745; }
            .model-competitive { background: #fff3cd; border-left: 4px solid #ffc107; }
            .model-lower { background: #f8d7da; border-left: 4px solid #dc3545; }
            .significance-yes { color: #28a745; font-weight: bold; }
            .significance-no { color: #6c757d; }
            .nav-header { background: #2c3e50; color: white; padding: 1rem 0; }
            .footer { background: #34495e; color: white; padding: 2rem 0; margin-top: 3rem; }
            .code-block { background: #f8f9fa; padding: 1rem; border-radius: 4px; border-left: 4px solid #007bff; }
        """)
    )
)

# Load results data
def load_evaluation_data():
    """Load the latest evaluation results."""
    data_loader = get_data_loader()
    return data_loader.load_latest_evaluation()

# Navigation component
def nav_header():
    return Nav(
        Div(
            A("Trans-Evals", href="/", style="font-size: 1.5rem; font-weight: bold; text-decoration: none; color: white;"),
            Ul(
                Li(A("Home", href="/")),
                Li(A("Results", href="/results")),
                Li(A("Methodology", href="/methodology")),
                Li(A("GitHub", href="https://github.com/rachaelroland/trans-evals", target="_blank")),
                style="list-style: none; display: flex; gap: 2rem; margin: 0;"
            ),
            style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto; padding: 0 1rem;"
        ),
        cls="nav-header"
    )

# Footer component  
def footer():
    return Footer(
        Div(
            Div(
                H4("Trans-Evals Project"),
                P("Evaluating language models for trans-inclusive language generation."),
                P("Framework for rigorous, peer-reviewed AI bias evaluation.")
            ),
            Div(
                H4("Links"),
                A("GitHub Repository", href="https://github.com/rachaelroland/trans-evals", target="_blank"),
                Br(),
                A("Research Paper", href="#", target="_blank"),
                Br(),
                A("Dataset Access", href="#"),
            ),
            Div(
                H4("Contact"),
                P("For questions about methodology or collaboration opportunities."),
                P(f"Last updated: {datetime.now().strftime('%B %Y')}")
            ),
            style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; max-width: 1200px; margin: 0 auto; padding: 0 1rem;"
        ),
        cls="footer"
    )

@rt("/")
def index():
    data = load_evaluation_data()
    
    return Html(
        nav_header(),
        
        # Hero section
        Section(
            Div(
                H1("Trans-Evals", style="font-size: 3rem; margin-bottom: 1rem;"),
                H2("Evaluating Language Models for Trans-Inclusive Language", style="font-size: 1.5rem; opacity: 0.9; font-weight: normal;"),
                P("A comprehensive framework for rigorous evaluation of AI language models on trans-inclusive language generation, with statistical analysis suitable for peer review.", style="font-size: 1.1rem; margin-top: 1.5rem; max-width: 800px;"),
                style="text-align: center; max-width: 1200px; margin: 0 auto; padding: 0 1rem;"
            ),
            cls="hero"
        ),
        
        # Main content
        Main(
            Div(
                Section(
                    H2("Key Findings"),
                    Div(
                        *[Div(
                            P(finding, style="margin: 0;"),
                            style="background: #f8f9fa; padding: 1rem; border-radius: 4px; border-left: 4px solid #007bff;"
                        ) for finding in data["key_findings"]],
                        style="display: grid; gap: 1rem;"
                    ) if "key_findings" in data else P("Loading findings...")
                ),
                
                Section(
                    H2("Model Performance Overview"),
                    P("Based on evaluation of 50 high-quality examples across 4 bias metrics:"),
                    Div(
                        *[Div(
                            H3(model_data["name"]),
                            P(model_data["assessment"], style="color: #666; font-style: italic;"),
                            Div(
                                Div(
                                    Span(f"{model_data['sentiment']['mean']:.3f}", cls="metric-score"),
                                    Div("Sentiment", cls="metric-label"),
                                    style="text-align: center;"
                                ),
                                Div(
                                    Span(f"{model_data['toxicity']['mean']:.3f}", cls="metric-score"),
                                    Div("Toxicity", cls="metric-label"),
                                    style="text-align: center;"
                                ),
                                Div(
                                    Span(f"{model_data['regard']['mean']:.3f}", cls="metric-score"),
                                    Div("Regard", cls="metric-label"),
                                    style="text-align: center;"
                                ),
                                Div(
                                    Span(f"{model_data['misgendering']['mean']:.3f}", cls="metric-score"),
                                    Div("Misgendering", cls="metric-label"),
                                    style="text-align: center;"
                                ),
                                style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;"
                            ),
                            cls=f"metric-card {model_data['class']}"
                        ) for model_id, model_data in data["models"].items()] if "models" in data else [P("Loading model data...")]
                    )
                ),
                
                Section(
                    H2("Dataset Innovation"),
                    Div(
                        Div(
                            H3("Comprehensive Coverage"),
                            P(f"Created {data['dataset_stats']['total_examples']} high-quality examples"),
                            Ul(
                                Li("Neo-pronoun support: xe/xem, ze/zir, ey/em, fae/faer"),
                                Li("Intersectional identities: race, disability, religion × trans identity"),
                                Li("Professional contexts across diverse industries"),
                                Li("WinoGender-style coreference tests")
                            )
                        ),
                        Div(
                            H3("Bias Type Distribution"),
                            *[P(f"{bias_type}: {percentage}") for bias_type, percentage in data["dataset_stats"]["bias_types"].items()]
                        ) if "dataset_stats" in data else P("Loading dataset statistics..."),
                        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;"
                    )
                ),
                
                Section(
                    H2("Methodology Highlights"),
                    Div(
                        Div(
                            H3("Statistical Rigor"),
                            Ul(
                                Li("Mann-Whitney U tests for statistical significance"),
                                Li("Cohen's d effect size calculations"),
                                Li("Bootstrap confidence intervals"),
                                Li("Fixed random seeds for reproducibility")
                            )
                        ),
                        Div(
                            H3("Quality Assurance"),
                            Ul(
                                Li("Text length filtering (20-500 characters)"),
                                Li("Content validation and deduplication"),
                                Li("Balanced sampling across bias types"),
                                Li("Comprehensive logging and error handling")
                            )
                        ),
                        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;"
                    )
                ),
                
                Section(
                    H2("Get Started"),
                    Div(
                        Div(
                            H3("For Researchers"),
                            P("Our framework provides peer-review ready evaluation with statistical rigor."),
                            A("View Detailed Results", href="/results", style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; text-decoration: none; border-radius: 4px; margin-top: 1rem;")
                        ),
                        Div(
                            H3("For Developers"),
                            P("Evaluate your models on trans-inclusive language with our open-source framework."),
                            A("GitHub Repository", href="https://github.com/rachaelroland/trans-evals", target="_blank", style="display: inline-block; background: #28a745; color: white; padding: 0.75rem 1.5rem; text-decoration: none; border-radius: 4px; margin-top: 1rem;")
                        ),
                        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;"
                    )
                ),
                
                style="max-width: 1200px; margin: 0 auto; padding: 2rem 1rem;"
            )
        ),
        
        footer()
    )

@rt("/results")
def results():
    data = load_evaluation_data()
    
    return Html(
        nav_header(),
        
        Main(
            Div(
                H1("Evaluation Results"),
                P("Comprehensive statistical analysis of language model performance on trans-inclusive language generation."),
                
                Section(
                    H2("Statistical Significance Testing"),
                    P("Pairwise comparisons using Mann-Whitney U tests (α = 0.05):"),
                    Div(
                        H3("Sentiment Analysis"),
                        Ul(
                            Li(Span("Claude Sonnet 4 vs Llama 3.3: ", style="font-weight: bold;"), Span("✅ Highly significant", cls="significance-yes"), " (p<0.001, large effect size: 2.40)"),
                            Li(Span("Kimi K2 vs Llama 3.3: ", style="font-weight: bold;"), Span("✅ Highly significant", cls="significance-yes"), " (p<0.001, large effect size: 1.57)"),
                            Li(Span("Claude Sonnet 4 vs Kimi K2: ", style="font-weight: bold;"), Span("✅ Significant", cls="significance-yes"), " (p=0.004, medium effect size: 0.66)")
                        ),
                        
                        H3("Regard (Respectful Treatment)"),
                        Ul(
                            Li(Span("Claude Sonnet 4 vs Llama 3.3: ", style="font-weight: bold;"), Span("✅ Significant", cls="significance-yes"), " (p=0.035, medium effect size: 0.54)"),
                            Li(Span("Kimi K2 vs Llama 3.3: ", style="font-weight: bold;"), Span("✅ Significant", cls="significance-yes"), " (p=0.028, medium effect size: 0.59)")
                        ),
                        
                        H3("Misgendering Detection"),
                        Ul(
                            Li(Span("All comparisons: ", style="font-weight: bold;"), Span("❌ Not significant", cls="significance-no"), " (all models ~0.43)")
                        ),
                        
                        H3("Toxicity Control"),
                        Ul(
                            Li("All models maintained excellent toxicity scores (<0.003)"),
                            Li(Span("Claude vs Llama: ", style="font-weight: bold;"), Span("✅ Significant", cls="significance-yes"), " (p<0.001)")
                        )
                    )
                ),
                
                Section(
                    H2("Detailed Model Comparison"),
                    Table(
                        Thead(
                            Tr(
                                Th("Model"),
                                Th("Misgendering ↑"),
                                Th("Toxicity ↓"),
                                Th("Sentiment ↑"),
                                Th("Regard ↑"),
                                Th("Assessment")
                            )
                        ),
                        Tbody(
                            *[Tr(
                                Td(model_data["name"]),
                                Td(f"{model_data['misgendering']['mean']:.3f}±{model_data['misgendering']['std']:.2f}"),
                                Td(f"{model_data['toxicity']['mean']:.3f}±{model_data['toxicity']['std']:.2f}"),
                                Td(f"{model_data['sentiment']['mean']:.3f}±{model_data['sentiment']['std']:.2f}"),
                                Td(f"{model_data['regard']['mean']:.3f}±{model_data['regard']['std']:.2f}"),
                                Td(model_data["assessment"])
                            ) for model_data in data["models"].values()] if "models" in data else []
                        )
                    ),
                    P("↑ = Higher is better, ↓ = Lower is better", style="font-style: italic; color: #666; margin-top: 1rem;")
                ),
                
                Section(
                    H2("Dataset Composition"),
                    Div(
                        Div(
                            H3("Identity Representation"),
                            Ul(
                                Li("Non-binary persons: 29% of examples"),
                                Li("Trans women: 14% of examples"),
                                Li("Trans men: 11% of examples"),
                                Li("Neo-pronoun users: 14% of examples"),
                                Li("Intersectional identities: 18% of examples")
                            )
                        ),
                        Div(
                            H3("Pronoun Coverage"),
                            P("Comprehensive support for:"),
                            Ul(
                                *[Li(pronoun) for pronoun in data["dataset_stats"]["pronoun_coverage"]] if "dataset_stats" in data else []
                            )
                        ),
                        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;"
                    )
                ),
                
                Section(
                    H2("Recommendations"),
                    Div(
                        Div(
                            H3("For Production Use"),
                            Ol(
                                Li(Strong("Recommended: "), "Claude Sonnet 4 for applications requiring high-quality trans-inclusive language"),
                                Li(Strong("Alternative: "), "Kimi K2 for cost-sensitive applications with acceptable trade-offs"),
                                Li(Strong("Monitoring: "), "Implement automated pronoun consistency checking regardless of model choice")
                            )
                        ),
                        Div(
                            H3("For Model Developers"),
                            Ol(
                                Li(Strong("Focus Area: "), "All models need improvement in misgendering detection (current ~0.43, target >0.8)"),
                                Li(Strong("Training Data: "), "Include more diverse trans narratives and neo-pronoun examples"),
                                Li(Strong("Evaluation: "), "Adopt comprehensive bias evaluation as standard practice")
                            )
                        ),
                        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 2rem;"
                    )
                ),
                
                style="max-width: 1200px; margin: 0 auto; padding: 2rem 1rem;"
            )
        ),
        
        footer()
    )

@rt("/methodology")
def methodology():
    return Html(
        nav_header(),
        
        Main(
            Div(
                H1("Methodology"),
                P("Detailed explanation of our evaluation framework and statistical methodology."),
                
                Section(
                    H2("Evaluation Framework"),
                    Div(
                        H3("Comprehensive Dataset Loader"),
                        P("Our framework combines multiple high-quality sources:"),
                        Ul(
                            Li("Synthetic trans-specific examples (186 created)"),
                            Li("WinoGender-style coreference tests adapted for trans identities"),
                            Li("Professional context examples across diverse industries"),
                            Li("Intersectional identity examples (race, disability, religion × trans)")
                        ),
                        cls="code-block"
                    )
                ),
                
                Section(
                    H2("Statistical Analysis"),
                    Div(
                        H3("Non-Parametric Testing"),
                        P("We use Mann-Whitney U tests for pairwise model comparisons because:"),
                        Ul(
                            Li("Makes no assumptions about data distribution"),
                            Li("Robust to outliers and non-normal data"),
                            Li("Appropriate for ordinal evaluation metrics")
                        ),
                        
                        H3("Effect Size Calculation"),
                        P("Cohen's d approximation provides practical significance beyond statistical significance:"),
                        Ul(
                            Li("Small effect: |d| < 0.2"),
                            Li("Medium effect: 0.2 ≤ |d| < 0.8"),
                            Li("Large effect: |d| ≥ 0.8")
                        ),
                        cls="code-block"
                    )
                ),
                
                Section(
                    H2("Quality Assurance"),
                    Div(
                        H3("Text Filtering"),
                        Ul(
                            Li("Minimum length: 20 characters for meaningful evaluation"),
                            Li("Maximum length: 500 characters to avoid complexity"),
                            Li("Template detection: Remove incomplete or placeholder text"),
                            Li("Deduplication: Ensure unique examples across all sources")
                        ),
                        
                        H3("Balanced Sampling"),
                        Ul(
                            Li("Stratified sampling across bias types"),
                            Li("Representative identity distribution"),
                            Li("Fixed random seeds for reproducibility"),
                            Li("Equal representation of pronoun types")
                        ),
                        cls="code-block"
                    )
                ),
                
                Section(
                    H2("Evaluation Metrics"),
                    Div(
                        Div(
                            H3("Misgendering Detection"),
                            P("Measures correct pronoun usage and consistency"),
                            Ul(
                                Li("Binary classification of pronoun correctness"),
                                Li("Handles traditional and neo-pronouns"),
                                Li("Context-aware evaluation")
                            )
                        ),
                        Div(
                            H3("Toxicity Analysis"),
                            P("Detects harmful or discriminatory language"),
                            Ul(
                                Li("Perspective API-based scoring"),
                                Li("Threshold: <0.1 excellent, <0.2 acceptable"),
                                Li("Trans-specific toxicity patterns")
                            )
                        ),
                        Div(
                            H3("Sentiment Analysis"),
                            P("Measures emotional tone and warmth"),
                            Ul(
                                Li("VADER sentiment analyzer"),
                                Li("Scale: 0-1 (negative to positive)"),
                                Li("Contextual sentiment evaluation")
                            )
                        ),
                        Div(
                            H3("Regard Assessment"),
                            P("Evaluates respectful treatment and dignity"),
                            Ul(
                                Li("Custom regard classification"),
                                Li("Binary positive/negative assessment"),
                                Li("Trans-inclusive respect metrics")
                            )
                        ),
                        style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;"
                    )
                ),
                
                Section(
                    H2("Reproducibility"),
                    Div(
                        H3("Open Source Framework"),
                        P("Complete codebase available on GitHub with:"),
                        Ul(
                            Li("Fixed random seeds for deterministic results"),
                            Li("Comprehensive logging and error handling"),
                            Li("Standardized evaluation pipeline"),
                            Li("Statistical analysis scripts")
                        ),
                        A("View Source Code", href="https://github.com/rachaelroland/trans-evals", target="_blank", style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; text-decoration: none; border-radius: 4px; margin-top: 1rem;"),
                        cls="code-block"
                    )
                ),
                
                style="max-width: 1200px; margin: 0 auto; padding: 2rem 1rem;"
            )
        ),
        
        footer()
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    serve(host="0.0.0.0", port=port)