# Trans-Evals Website

A FastHTML website showcasing the comprehensive evaluation of language models for trans-inclusive language generation.

## Features

- **Home Page**: Project overview, key findings, and methodology highlights
- **Results Page**: Detailed statistical analysis and model comparisons  
- **Methodology Page**: In-depth explanation of evaluation framework
- **Real Data Integration**: Loads actual evaluation results from JSON and markdown reports
- **Responsive Design**: Clean, professional styling with PicoCSS

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
python app.py

# Visit http://localhost:8000
```

## Deployment

### Render Deployment

This website is configured for deployment on Render via git push:

1. Connect your GitHub repository to Render
2. Use the `render.yaml` configuration file
3. Environment variables:
   - `PORT`: Automatically set by Render
4. Build command: `pip install -r requirements.txt`
5. Start command: `python app.py`

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PORT=8000

# Run in production
python app.py
```

## Architecture

- **`app.py`**: Main FastHTML application with routes and UI components
- **`data_loader.py`**: Loads and processes evaluation results from project files
- **`requirements.txt`**: Python dependencies for deployment
- **`render.yaml`**: Render platform configuration

## Data Sources

The website automatically loads data from:
- `../reports/statistical_evaluation_*.md`: Statistical analysis reports
- `../tests/results/*_statistical_*.json`: Individual model evaluation results
- `../reports/comprehensive_evaluation_summary_*.md`: Overview reports

## Routes

- `/` - Home page with project overview
- `/results` - Detailed evaluation results and statistical analysis
- `/methodology` - Framework explanation and methodology details

## Styling

Uses PicoCSS for clean, semantic styling with custom CSS for:
- Model performance cards with color coding
- Statistical significance indicators  
- Hero sections and navigation
- Responsive grid layouts

## Updates

The website automatically reflects new evaluation results by:
1. Reading the latest reports and JSON files
2. Parsing statistical significance data
3. Updating model performance metrics
4. Refreshing visualizations and summaries

For manual updates, simply push new evaluation results to the repository and redeploy.