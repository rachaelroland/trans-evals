"""
FastHTML website for trans-evals project with dark theme.
"""

from fasthtml.common import *
from data_loader import get_data_loader
from datetime import datetime

# Initialize FastHTML app
app, rt = fast_app(
    title="Trans-Evals: Evaluating Language Models for Trans-Inclusive Language",
    hdrs=(
        # Tailwind CSS CDN
        Script(src="https://cdn.tailwindcss.com"),
        # Custom dark theme styles
        Style("""
            body {
                background-color: #0d1117;
                color: #c9d1d9;
            }
            .dark-card {
                background-color: #161b22;
                border-color: #30363d;
            }
            .dark-nav {
                background-color: #161b22;
                border-bottom: 1px solid #30363d;
            }
            .gradient-text {
                background: linear-gradient(to right, #a78bfa, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .hero-gradient {
                background: linear-gradient(to bottom, rgba(139, 92, 246, 0.1), transparent);
            }
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
            A("Trans-Evals", href="/", style="font-size: 1.5rem; font-weight: bold; color: white; text-decoration: none;"),
            Ul(
                Li(A("Home", href="/", style="color: #8b949e; text-decoration: none;")),
                Li(A("Results", href="/results", style="color: #8b949e; text-decoration: none;")),
                Li(A("Methodology", href="/methodology", style="color: #8b949e; text-decoration: none;")),
                Li(A("GitHub", href="https://github.com/rachaelroland/trans-evals", target="_blank", style="color: #8b949e; text-decoration: none;")),
                style="list-style: none; display: flex; gap: 2rem; margin: 0;"
            ),
            style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto; padding: 1rem;"
        ),
        cls="dark-nav"
    )

# Footer component
def footer():
    return Footer(
        Div(
            Div(
                H4("Trans-Evals Project", style="color: white; font-size: 1.25rem; margin-bottom: 0.5rem;"),
                P("Evaluating language models for trans-inclusive language generation.", style="color: #8b949e;"),
                P("Framework for rigorous, peer-reviewed AI bias evaluation.", style="color: #8b949e;")
            ),
            Div(
                H4("Links", style="color: white; font-size: 1.25rem; margin-bottom: 0.5rem;"),
                A("GitHub Repository", href="https://github.com/rachaelroland/trans-evals", target="_blank", style="color: #58a6ff; text-decoration: none; display: block; margin-bottom: 0.5rem;"),
                A("Research Paper", href="#", target="_blank", style="color: #58a6ff; text-decoration: none; display: block; margin-bottom: 0.5rem;"),
                A("Dataset Access", href="#", style="color: #58a6ff; text-decoration: none; display: block;"),
            ),
            Div(
                H4("Contact", style="color: white; font-size: 1.25rem; margin-bottom: 0.5rem;"),
                P("For questions about methodology or collaboration opportunities.", style="color: #8b949e;"),
                P(f"Last updated: {datetime.now().strftime('%B %Y')}", style="color: #8b949e;")
            ),
            style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; max-width: 1200px; margin: 0 auto; padding: 2rem 1rem;"
        ),
        style="background-color: #161b22; padding: 2rem 0; margin-top: 4rem; border-top: 1px solid #30363d;"
    )

@rt("/")
def index():
    data = load_evaluation_data()
    
    return Html(
        Body(
            nav_header(),
            
            # Hero section
            Section(
                Div(
                    H1("Trans-Evals", cls="gradient-text", style="font-size: 4rem; font-weight: bold; margin-bottom: 1rem;"),
                    H2("Evaluating Language Models for Trans-Inclusive Language", style="font-size: 1.5rem; color: #c9d1d9; font-weight: normal; margin-bottom: 1.5rem;"),
                    P("A comprehensive framework for rigorous evaluation of AI language models on trans-inclusive language generation, with statistical analysis suitable for peer review.", 
                      style="font-size: 1.125rem; color: #8b949e; max-width: 800px; margin: 0 auto;"),
                    style="text-align: center; max-width: 1200px; margin: 0 auto; padding: 0 1rem;"
                ),
                cls="hero-gradient",
                style="padding: 4rem 0; border-bottom: 1px solid #30363d;"
            ),
            
            # Main content
            Main(
                Div(
                    # Key Findings
                    Section(
                        H2("Key Findings", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            *[Div(
                                P(finding, style="color: #c9d1d9; margin: 0;"),
                                style="background-color: #161b22; padding: 1rem; border-radius: 0.5rem; border: 1px solid #30363d; border-left: 4px solid #58a6ff;"
                            ) for finding in data["key_findings"]],
                            style="display: grid; gap: 1rem;"
                        ) if "key_findings" in data else P("Loading findings...", style="color: #8b949e;"),
                        style="margin-bottom: 3rem;"
                    ),
                    
                    # Model Performance
                    Section(
                        H2("Model Performance Overview", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 0.5rem;"),
                        P("Based on evaluation of 50 high-quality examples across 4 bias metrics:", style="color: #8b949e; margin-bottom: 1.5rem;"),
                        Div(
                            *[Div(
                                H3(model_data["name"], style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.5rem;"),
                                P(model_data["assessment"], style="color: #8b949e; font-style: italic; margin-bottom: 1rem;"),
                                Div(
                                    Div(
                                        Span(f"{model_data['sentiment']['mean']:.3f}", style="font-size: 2rem; font-weight: bold; color: white; display: block;"),
                                        Div("Sentiment", style="font-size: 0.75rem; text-transform: uppercase; color: #6e7681; margin-top: 0.25rem;"),
                                        style="text-align: center;"
                                    ),
                                    Div(
                                        Span(f"{model_data['toxicity']['mean']:.3f}", style="font-size: 2rem; font-weight: bold; color: white; display: block;"),
                                        Div("Toxicity", style="font-size: 0.75rem; text-transform: uppercase; color: #6e7681; margin-top: 0.25rem;"),
                                        style="text-align: center;"
                                    ),
                                    Div(
                                        Span(f"{model_data['regard']['mean']:.3f}", style="font-size: 2rem; font-weight: bold; color: white; display: block;"),
                                        Div("Regard", style="font-size: 0.75rem; text-transform: uppercase; color: #6e7681; margin-top: 0.25rem;"),
                                        style="text-align: center;"
                                    ),
                                    Div(
                                        Span(f"{model_data['misgendering']['mean']:.3f}", style="font-size: 2rem; font-weight: bold; color: white; display: block;"),
                                        Div("Misgendering", style="font-size: 0.75rem; text-transform: uppercase; color: #6e7681; margin-top: 0.25rem;"),
                                        style="text-align: center;"
                                    ),
                                    style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 1rem;"
                                ),
                                style=f"background-color: #161b22; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid {
                                    '#10b981' if 'winner' in model_data['class'] else 
                                    '#f59e0b' if 'competitive' in model_data['class'] else 
                                    '#ef4444'
                                }; transition: transform 0.2s;"
                            ) for model_id, model_data in data["models"].items()] if "models" in data else [P("Loading model data...", style="color: #8b949e;")],
                            style="display: grid; gap: 1.5rem;"
                        ),
                        style="margin-bottom: 3rem;"
                    ),
                    
                    # Dataset Innovation
                    Section(
                        H2("Dataset Innovation", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            Div(
                                H3("Comprehensive Coverage", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                                P(f"Created {data['dataset_stats']['total_examples']} high-quality examples", style="color: #c9d1d9; margin-bottom: 0.5rem;"),
                                Ul(
                                    Li("Neo-pronoun support: xe/xem, ze/zir, ey/em, fae/faer", style="color: #8b949e;"),
                                    Li("Intersectional identities: race, disability, religion × trans identity", style="color: #8b949e;"),
                                    Li("Professional contexts across diverse industries", style="color: #8b949e;"),
                                    Li("WinoGender-style coreference tests", style="color: #8b949e;"),
                                    style="list-style: disc; list-style-position: inside; margin-left: 1rem;"
                                )
                            ),
                            Div(
                                H3("Bias Type Distribution", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                                *[P(f"{bias_type}: {percentage}", style="color: #8b949e; margin-bottom: 0.25rem;") 
                                  for bias_type, percentage in data["dataset_stats"]["bias_types"].items()]
                            ) if "dataset_stats" in data else P("Loading dataset statistics...", style="color: #8b949e;"),
                            style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;"
                        ),
                        style="margin-bottom: 3rem;"
                    ),
                    
                    # Methodology
                    Section(
                        H2("Methodology Highlights", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            Div(
                                H3("Statistical Rigor", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                                Ul(
                                    Li("Mann-Whitney U tests for statistical significance", style="color: #8b949e;"),
                                    Li("Cohen's d effect size calculations", style="color: #8b949e;"),
                                    Li("Bootstrap confidence intervals", style="color: #8b949e;"),
                                    Li("Fixed random seeds for reproducibility", style="color: #8b949e;"),
                                    style="list-style: disc; list-style-position: inside; margin-left: 1rem;"
                                )
                            ),
                            Div(
                                H3("Quality Assurance", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                                Ul(
                                    Li("Text length filtering (20-500 characters)", style="color: #8b949e;"),
                                    Li("Content validation and deduplication", style="color: #8b949e;"),
                                    Li("Balanced sampling across bias types", style="color: #8b949e;"),
                                    Li("Comprehensive logging and error handling", style="color: #8b949e;"),
                                    style="list-style: disc; list-style-position: inside; margin-left: 1rem;"
                                )
                            ),
                            style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;"
                        ),
                        style="margin-bottom: 3rem;"
                    ),
                    
                    # Get Started
                    Section(
                        H2("Get Started", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            Div(
                                H3("For Researchers", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.5rem;"),
                                P("Our framework provides peer-review ready evaluation with statistical rigor.", style="color: #8b949e; margin-bottom: 1rem;"),
                                A("View Detailed Results", href="/results", 
                                  style="display: inline-block; background-color: #238636; color: white; font-weight: 600; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none;")
                            ),
                            Div(
                                H3("For Developers", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.5rem;"),
                                P("Evaluate your models on trans-inclusive language with our open-source framework.", style="color: #8b949e; margin-bottom: 1rem;"),
                                A("GitHub Repository", href="https://github.com/rachaelroland/trans-evals", target="_blank",
                                  style="display: inline-block; background-color: #1f6feb; color: white; font-weight: 600; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none;")
                            ),
                            style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;"
                        )
                    ),
                    
                    style="max-width: 1200px; margin: 0 auto; padding: 2rem 1rem;"
                )
            ),
            
            footer(),
            style="background-color: #0d1117; color: #c9d1d9; min-height: 100vh;"
        )
    )

@rt("/results")
def results():
    data = load_evaluation_data()
    
    return Html(
        Body(
            nav_header(),
            
            Main(
                Div(
                    H1("Evaluation Results", style="font-size: 2.5rem; font-weight: bold; color: white; margin-bottom: 0.5rem;"),
                    P("Comprehensive statistical analysis of language model performance on trans-inclusive language generation.", 
                      style="font-size: 1.125rem; color: #8b949e; margin-bottom: 2rem;"),
                    
                    # Statistical Significance
                    Section(
                        H2("Statistical Significance Testing", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1rem;"),
                        P("Pairwise comparisons using Mann-Whitney U tests (α = 0.05):", style="color: #8b949e; margin-bottom: 1.5rem;"),
                        Div(
                            H3("Sentiment Analysis", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem;"),
                            Ul(
                                Li(
                                    Span("Claude Sonnet 4 vs Llama 3.3: ", style="font-weight: 600; color: #c9d1d9;"),
                                    Span("✅ Highly significant", style="color: #10b981; font-weight: bold;"),
                                    " (p<0.001, large effect size: 2.40)",
                                    style="color: #8b949e; margin-bottom: 0.5rem;"
                                ),
                                Li(
                                    Span("Kimi K2 vs Llama 3.3: ", style="font-weight: 600; color: #c9d1d9;"),
                                    Span("✅ Highly significant", style="color: #10b981; font-weight: bold;"),
                                    " (p<0.001, large effect size: 1.57)",
                                    style="color: #8b949e; margin-bottom: 0.5rem;"
                                ),
                                Li(
                                    Span("Claude Sonnet 4 vs Kimi K2: ", style="font-weight: 600; color: #c9d1d9;"),
                                    Span("✅ Significant", style="color: #10b981; font-weight: bold;"),
                                    " (p=0.004, medium effect size: 0.66)",
                                    style="color: #8b949e;"
                                ),
                                style="list-style: none; margin-left: 1rem;"
                            ),
                            
                            H3("Regard (Respectful Treatment)", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem; margin-top: 1.5rem;"),
                            Ul(
                                Li(
                                    Span("Claude Sonnet 4 vs Llama 3.3: ", style="font-weight: 600; color: #c9d1d9;"),
                                    Span("✅ Significant", style="color: #10b981; font-weight: bold;"),
                                    " (p=0.035, medium effect size: 0.54)",
                                    style="color: #8b949e; margin-bottom: 0.5rem;"
                                ),
                                Li(
                                    Span("Kimi K2 vs Llama 3.3: ", style="font-weight: 600; color: #c9d1d9;"),
                                    Span("✅ Significant", style="color: #10b981; font-weight: bold;"),
                                    " (p=0.028, medium effect size: 0.59)",
                                    style="color: #8b949e;"
                                ),
                                style="list-style: none; margin-left: 1rem;"
                            ),
                            
                            H3("Misgendering Detection", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem; margin-top: 1.5rem;"),
                            Ul(
                                Li(
                                    Span("All comparisons: ", style="font-weight: 600; color: #c9d1d9;"),
                                    Span("❌ Not significant", style="color: #6e7681;"),
                                    " (all models ~0.43)",
                                    style="color: #8b949e;"
                                ),
                                style="list-style: none; margin-left: 1rem;"
                            ),
                            
                            H3("Toxicity Control", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem; margin-top: 1.5rem;"),
                            Ul(
                                Li("All models maintained excellent toxicity scores (<0.003)", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                Li(
                                    Span("Claude vs Llama: ", style="font-weight: 600; color: #c9d1d9;"),
                                    Span("✅ Significant", style="color: #10b981; font-weight: bold;"),
                                    " (p<0.001)",
                                    style="color: #8b949e;"
                                ),
                                style="list-style: none; margin-left: 1rem;"
                            ),
                            style="background-color: #161b22; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                        ),
                        style="margin-bottom: 3rem;"
                    ),
                    
                    # Model Comparison Table
                    Section(
                        H2("Detailed Model Comparison", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            Table(
                                Thead(
                                    Tr(
                                        Th("Model", style="padding: 0.75rem 1.5rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;"),
                                        Th("Misgendering ↑", style="padding: 0.75rem 1.5rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;"),
                                        Th("Toxicity ↓", style="padding: 0.75rem 1.5rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;"),
                                        Th("Sentiment ↑", style="padding: 0.75rem 1.5rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;"),
                                        Th("Regard ↑", style="padding: 0.75rem 1.5rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;"),
                                        Th("Assessment", style="padding: 0.75rem 1.5rem; text-align: left; font-size: 0.75rem; font-weight: 500; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em;"),
                                        style="background-color: #21262d;"
                                    )
                                ),
                                Tbody(
                                    *[Tr(
                                        Td(model_data["name"], style="padding: 1rem 1.5rem; font-size: 0.875rem; color: #c9d1d9;"),
                                        Td(f"{model_data['misgendering']['mean']:.3f}±{model_data['misgendering']['std']:.2f}", 
                                           style="padding: 1rem 1.5rem; font-size: 0.875rem; color: #c9d1d9;"),
                                        Td(f"{model_data['toxicity']['mean']:.3f}±{model_data['toxicity']['std']:.2f}", 
                                           style="padding: 1rem 1.5rem; font-size: 0.875rem; color: #c9d1d9;"),
                                        Td(f"{model_data['sentiment']['mean']:.3f}±{model_data['sentiment']['std']:.2f}", 
                                           style="padding: 1rem 1.5rem; font-size: 0.875rem; color: #c9d1d9;"),
                                        Td(f"{model_data['regard']['mean']:.3f}±{model_data['regard']['std']:.2f}", 
                                           style="padding: 1rem 1.5rem; font-size: 0.875rem; color: #c9d1d9;"),
                                        Td(model_data["assessment"], style="padding: 1rem 1.5rem; font-size: 0.875rem; color: #c9d1d9;"),
                                        style="border-top: 1px solid #30363d;"
                                    ) for model_data in data["models"].values()] if "models" in data else [],
                                    style="background-color: #161b22;"
                                ),
                                style="width: 100%; border-collapse: collapse;"
                            ),
                            style="overflow-x: auto; background-color: #161b22; border-radius: 0.5rem; border: 1px solid #30363d;"
                        ),
                        P("↑ = Higher is better, ↓ = Lower is better", style="font-size: 0.875rem; font-style: italic; color: #6e7681; margin-top: 0.5rem;"),
                        style="margin-bottom: 3rem;"
                    ),
                    
                    # Recommendations
                    Section(
                        H2("Recommendations", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            Div(
                                H3("For Production Use", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                                Ol(
                                    Li(Strong("Recommended: ", style="color: #c9d1d9;"), "Claude Sonnet 4 for applications requiring high-quality trans-inclusive language", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                    Li(Strong("Alternative: ", style="color: #c9d1d9;"), "Kimi K2 for cost-sensitive applications with acceptable trade-offs", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                    Li(Strong("Monitoring: ", style="color: #c9d1d9;"), "Implement automated pronoun consistency checking regardless of model choice", style="color: #8b949e;"),
                                    style="list-style: decimal; list-style-position: inside; margin-left: 1rem;"
                                )
                            ),
                            Div(
                                H3("For Model Developers", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                                Ol(
                                    Li(Strong("Focus Area: ", style="color: #c9d1d9;"), "All models need improvement in misgendering detection (current ~0.43, target >0.8)", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                    Li(Strong("Training Data: ", style="color: #c9d1d9;"), "Include more diverse trans narratives and neo-pronoun examples", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                    Li(Strong("Evaluation: ", style="color: #c9d1d9;"), "Adopt comprehensive bias evaluation as standard practice", style="color: #8b949e;"),
                                    style="list-style: decimal; list-style-position: inside; margin-left: 1rem;"
                                )
                            ),
                            style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 2rem;"
                        )
                    ),
                    
                    style="max-width: 1200px; margin: 0 auto; padding: 2rem 1rem;"
                )
            ),
            
            footer(),
            style="background-color: #0d1117; color: #c9d1d9; min-height: 100vh;"
        )
    )

@rt("/methodology")
def methodology():
    return Html(
        Body(
            nav_header(),
            
            Main(
                Div(
                    H1("Methodology", style="font-size: 2.5rem; font-weight: bold; color: white; margin-bottom: 0.5rem;"),
                    P("Detailed explanation of our evaluation framework and statistical methodology.", 
                      style="font-size: 1.125rem; color: #8b949e; margin-bottom: 2rem;"),
                    
                    # Evaluation Framework
                    Section(
                        H2("Evaluation Framework", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            H3("Comprehensive Dataset Loader", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                            P("Our framework combines multiple high-quality sources:", style="color: #8b949e; margin-bottom: 0.75rem;"),
                            Ul(
                                Li("Synthetic trans-specific examples (186 created)", style="color: #8b949e;"),
                                Li("WinoGender-style coreference tests adapted for trans identities", style="color: #8b949e;"),
                                Li("Professional context examples across diverse industries", style="color: #8b949e;"),
                                Li("Intersectional identity examples (race, disability, religion × trans)", style="color: #8b949e;"),
                                style="list-style: disc; list-style-position: inside; margin-left: 1rem;"
                            ),
                            style="background-color: #161b22; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                        ),
                        style="margin-bottom: 2rem;"
                    ),
                    
                    # Statistical Analysis
                    Section(
                        H2("Statistical Analysis", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            H3("Non-Parametric Testing", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                            P("We use Mann-Whitney U tests for pairwise model comparisons because:", style="color: #8b949e; margin-bottom: 0.75rem;"),
                            Ul(
                                Li("Makes no assumptions about data distribution", style="color: #8b949e;"),
                                Li("Robust to outliers and non-normal data", style="color: #8b949e;"),
                                Li("Appropriate for ordinal evaluation metrics", style="color: #8b949e;"),
                                style="list-style: disc; list-style-position: inside; margin-left: 1rem;"
                            ),
                            
                            H3("Effect Size Calculation", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem; margin-top: 1.5rem;"),
                            P("Cohen's d approximation provides practical significance beyond statistical significance:", style="color: #8b949e; margin-bottom: 0.75rem;"),
                            Ul(
                                Li("Small effect: |d| < 0.2", style="color: #8b949e;"),
                                Li("Medium effect: 0.2 ≤ |d| < 0.8", style="color: #8b949e;"),
                                Li("Large effect: |d| ≥ 0.8", style="color: #8b949e;"),
                                style="list-style: disc; list-style-position: inside; margin-left: 1rem;"
                            ),
                            style="background-color: #161b22; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                        ),
                        style="margin-bottom: 2rem;"
                    ),
                    
                    # Evaluation Metrics
                    Section(
                        H2("Evaluation Metrics", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            Div(
                                H3("Misgendering Detection", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem;"),
                                P("Measures correct pronoun usage and consistency", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                Ul(
                                    Li("Binary classification of pronoun correctness", style="color: #8b949e;"),
                                    Li("Handles traditional and neo-pronouns", style="color: #8b949e;"),
                                    Li("Context-aware evaluation", style="color: #8b949e;"),
                                    style="list-style: disc; list-style-position: inside; margin-left: 1rem; font-size: 0.875rem;"
                                ),
                                style="background-color: #21262d; padding: 1rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                            ),
                            Div(
                                H3("Toxicity Analysis", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem;"),
                                P("Detects harmful or discriminatory language", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                Ul(
                                    Li("Perspective API-based scoring", style="color: #8b949e;"),
                                    Li("Threshold: <0.1 excellent, <0.2 acceptable", style="color: #8b949e;"),
                                    Li("Trans-specific toxicity patterns", style="color: #8b949e;"),
                                    style="list-style: disc; list-style-position: inside; margin-left: 1rem; font-size: 0.875rem;"
                                ),
                                style="background-color: #21262d; padding: 1rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                            ),
                            Div(
                                H3("Sentiment Analysis", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem;"),
                                P("Measures emotional tone and warmth", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                Ul(
                                    Li("VADER sentiment analyzer", style="color: #8b949e;"),
                                    Li("Scale: 0-1 (negative to positive)", style="color: #8b949e;"),
                                    Li("Contextual sentiment evaluation", style="color: #8b949e;"),
                                    style="list-style: disc; list-style-position: inside; margin-left: 1rem; font-size: 0.875rem;"
                                ),
                                style="background-color: #21262d; padding: 1rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                            ),
                            Div(
                                H3("Regard Assessment", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.75rem;"),
                                P("Evaluates respectful treatment and dignity", style="color: #8b949e; margin-bottom: 0.5rem;"),
                                Ul(
                                    Li("Custom regard classification", style="color: #8b949e;"),
                                    Li("Binary positive/negative assessment", style="color: #8b949e;"),
                                    Li("Trans-inclusive respect metrics", style="color: #8b949e;"),
                                    style="list-style: disc; list-style-position: inside; margin-left: 1rem; font-size: 0.875rem;"
                                ),
                                style="background-color: #21262d; padding: 1rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                            ),
                            style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;"
                        ),
                        style="margin-bottom: 2rem;"
                    ),
                    
                    # Reproducibility
                    Section(
                        H2("Reproducibility", style="font-size: 2rem; font-weight: bold; color: white; margin-bottom: 1.5rem;"),
                        Div(
                            H3("Open Source Framework", style="font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 1rem;"),
                            P("Complete codebase available on GitHub with:", style="color: #8b949e; margin-bottom: 0.75rem;"),
                            Ul(
                                Li("Fixed random seeds for deterministic results", style="color: #8b949e;"),
                                Li("Comprehensive logging and error handling", style="color: #8b949e;"),
                                Li("Standardized evaluation pipeline", style="color: #8b949e;"),
                                Li("Statistical analysis scripts", style="color: #8b949e;"),
                                style="list-style: disc; list-style-position: inside; margin-left: 1rem; margin-bottom: 1rem;"
                            ),
                            A("View Source Code", href="https://github.com/rachaelroland/trans-evals", target="_blank",
                              style="display: inline-block; background-color: #1f6feb; color: white; font-weight: 600; padding: 0.75rem 1.5rem; border-radius: 0.5rem; text-decoration: none;"),
                            style="background-color: #161b22; padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #30363d;"
                        )
                    ),
                    
                    style="max-width: 1200px; margin: 0 auto; padding: 2rem 1rem;"
                )
            ),
            
            footer(),
            style="background-color: #0d1117; color: #c9d1d9; min-height: 100vh;"
        )
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    serve(host="0.0.0.0", port=port)