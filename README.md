# BabyLM Training and Evaluation Project

This project implements and compares two different BabyLM (Baby Language Model) architectures for a university assignment on language model training.

## Project Structure

```
├── model.py              # Transformer model implementation
├── tokenizer.py          # Simple tokenizer implementation
├── train.py              # Training script for both models
├── evaluate.py           # Evaluation script using minimal pairs
├── generate_report.py    # Automatic LaTeX report generation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Models

Two transformer-based language models are implemented:

### Small BabyLM
- **Embedding dimension**: 64
- **Attention heads**: 2
- **Layers**: 2
- **Feed-forward dimension**: 256
- **Max sequence length**: 256

### Large BabyLM
- **Embedding dimension**: 128
- **Attention heads**: 4
- **Layers**: 4
- **Feed-forward dimension**: 512
- **Max sequence length**: 512

## Installation

### Option 1: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv babylm_env

# Activate virtual environment
source babylm_env/bin/activate  # Linux/Mac
# OR on Windows: babylm_env\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Direct Installation

```bash
pip install -r requirements.txt  # May require --user flag on some systems
```

**Note**: If you get "externally-managed-environment" error, use Option 1 (virtual environment).

## Usage

### 1. Train Both Models

```bash
python train.py
# Note: Use 'python3' instead of 'python' if needed on your system
```

This will:
- Create a simple tokenizer with 500 vocabulary size
- Train both small and large models for 3 epochs
- Save models to `models/small_babylm/` and `models/large_babylm/`
- Generate training loss curves

### 2. Evaluate Models

```bash
python evaluate.py
# Note: Use 'python3' instead of 'python' if needed on your system
```

This will:
- Load both trained models
- Create 1000+ minimal pairs for evaluation
- Test models on grammatical vs. ungrammatical sentences
- Generate comparison results and plots

### 3. Generate Report

```bash
python generate_report.py
# Note: Use 'python3' instead of 'python' if needed on your system
```

This will:
- Create a LaTeX report based on evaluation results
- Automatically compile to PDF (if pdflatex is available)
- Include model configurations, results, and analysis

## Key Features

- **From-scratch implementation**: Complete transformer implementation without using pre-built model classes
- **Minimal pairs evaluation**: Tests grammatical competence on 1000+ sentence pairs
- **Automatic reporting**: Generates professional LaTeX/PDF reports
- **Resource-efficient**: Works with limited computational resources
- **Extensible**: Easy to modify architectures and training parameters

## Evaluation Methodology

Models are evaluated on minimal pairs testing:
- Subject-verb agreement
- Auxiliary verb agreement  
- Word order preferences
- Determiner-noun agreement

Accuracy is measured by whether models assign lower perplexity to grammatically correct sentences.

## Expected Results

The larger model typically shows improved performance on grammatical tasks due to:
- Increased model capacity
- Better attention mechanisms
- Deeper linguistic representations

## File Outputs

After running the complete pipeline:
- `models/`: Trained model checkpoints and configurations
- `training_curves.png`: Training loss visualization
- `model_comparison.png`: Evaluation results comparison
- `evaluation_results.json`: Detailed evaluation data
- `babylm_report.tex`: LaTeX report source
- `babylm_report.pdf`: Final PDF report

## Customization

You can easily modify:
- Model architectures in `model.py`
- Training data in `train.py` (replace `prepare_sample_data()`)
- Evaluation criteria in `evaluate.py`
- Report template in `generate_report.py`

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Basic scientific Python stack (numpy, matplotlib, tqdm)
- Optional: pdflatex for report compilation

## License

This project is created for educational purposes as part of a university assignment. 