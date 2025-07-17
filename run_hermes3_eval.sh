#!/bin/bash

# Hermes 3 Dataset Professional Evaluation Script
# This script runs a complete evaluation of the Hermes 3 dataset

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          Hermes 3 Dataset Professional Evaluation             ║"
echo "║                                                               ║"
echo "║  This will download and evaluate the Hermes 3 dataset         ║"
echo "║  for trans-inclusive language bias using Claude 3.5           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY environment variable not set"
    echo ""
    echo "Please set your API key:"
    echo "  export OPENROUTER_API_KEY=your_key_here"
    echo ""
    exit 1
fi

echo "✅ API key detected"
echo ""

# Create directories
mkdir -p data/hermes3
mkdir -p hermes3_evaluation_results

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install datasets huggingface-hub pyarrow sqlalchemy aiohttp > /dev/null 2>&1

# Set evaluation parameters
SAMPLE_SIZE=500
CONFIDENCE=0.95

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        --quick)
            SAMPLE_SIZE=100
            shift
            ;;
        --thorough)
            SAMPLE_SIZE=1000
            CONFIDENCE=0.99
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "📊 Evaluation parameters:"
echo "   Sample size: $SAMPLE_SIZE"
echo "   Confidence level: $CONFIDENCE"
echo ""

# Estimate cost
COST_MIN=$(echo "scale=2; $SAMPLE_SIZE * 0.022" | bc)
COST_MAX=$(echo "scale=2; $SAMPLE_SIZE * 0.030" | bc)
echo "💰 Estimated cost: \$$COST_MIN - \$$COST_MAX USD"
echo ""

# Confirm before proceeding
read -p "🤔 Do you want to proceed? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting evaluation..."
echo ""

# Run the evaluation
python examples/download_and_eval_hermes3.py \
    --sample-size $SAMPLE_SIZE \
    --confidence $CONFIDENCE \
    --output-dir hermes3_evaluation_results

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "📄 Results are available in:"
    echo "   - Report: hermes3_evaluation_results/professional_evaluation_*.md"
    echo "   - Database: hermes3_evaluation_results/hermes3_professional_eval.db"
    echo "   - Metadata: hermes3_evaluation_results/hermes3_evaluation_metadata_*.json"
    echo ""
    
    # Show the report location
    REPORT=$(ls -t hermes3_evaluation_results/professional_evaluation_*.md 2>/dev/null | head -1)
    if [ -n "$REPORT" ]; then
        echo "📖 To view the report:"
        echo "   cat $REPORT"
        echo ""
        echo "Or open in your preferred markdown viewer."
    fi
else
    echo ""
    echo "❌ Evaluation failed. Please check the error messages above."
    exit 1
fi