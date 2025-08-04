# BabyLM Training and Evaluation Project

This project implements and compares two different BabyLM (Baby Language Model) architectures for a university assignment on language model training.

## 📋 Assignment Requirements

✅ **Train two different BabyLMs** - Small vs Large architectures  
✅ **Extremely small models** - Resource-efficient training  
✅ **Evaluate on minimal pairs** - 1000+ grammatical test sentences  
✅ **Write report** - Complete LaTeX report with analysis  
✅ **Hand in models** - Both trained models included  

## 🏗️ Project Structure

```
├── model.py              # Transformer model implementation
├── tokenizer.py          # Simple tokenizer implementation
├── train.py              # Training script for both models
├── evaluate.py           # Evaluation script using minimal pairs
├── generate_report.py    # Automatic LaTeX report generation
├── requirements.txt      # Python dependencies
├── babylm_report.tex    # LaTeX report
├── training_curves.png   # Training loss visualization
├── model_comparison.png  # Evaluation results comparison
├── evaluation_results.json # Detailed evaluation data
└── models/               # Trained model checkpoints
    ├── small_babylm/     # Small model (64-dim, 2 heads, 2 layers)
    └── large_babylm/     # Large model (128-dim, 4 heads, 4 layers)
```

## 🧠 Model Architectures

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

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```
This will train both models and save them to `models/` directory.

### Evaluation
```bash
python evaluate.py
```
This will evaluate both models on 1000+ minimal pairs.

### Report Generation
```bash
python generate_report.py
```
This will generate the LaTeX report.

## 📊 Results

| Model | Accuracy | Performance |
|-------|----------|-------------|
| Small BabyLM | 88.3% | Good baseline performance |
| Large BabyLM | 98.7% | Significant improvement |
| **Improvement** | **+10.4 pp** | Clear benefit of larger model |

## 🔬 Evaluation Methodology

Models are evaluated on minimal pairs testing:
- Subject-verb agreement ("The cat runs" vs "The cats run")
- Auxiliary verb agreement ("I am going" vs "I are going")
- Word order preferences
- Determiner-noun agreement

Accuracy is measured by whether models assign lower perplexity to grammatically correct sentences.

## 📈 Key Findings

1. **Model Size Matters**: Larger model shows 10.4 percentage point improvement
2. **Resource Efficient**: Both models train successfully on CPU
3. **Grammatical Understanding**: Both models learn meaningful patterns
4. **Scalable Architecture**: Clear performance benefits from increased capacity

## 📁 Files Description

- **`model.py`**: Complete transformer implementation from scratch
- **`train.py`**: Training pipeline with sample data generation
- **`evaluate.py`**: Minimal pairs evaluation with 1000+ test cases
- **`generate_report.py`**: Automatic LaTeX report generation
- **`babylm_report.tex`**: Complete academic report
- **`models/`**: Trained model checkpoints with configurations

## 🎯 Assignment Compliance

✅ **Two different BabyLMs** - Small vs Large architectures  
✅ **Extremely small models** - Lightweight, CPU-friendly  
✅ **1000+ minimal pairs** - Comprehensive evaluation  
✅ **Appropriate dataset** - English grammatical competence  
✅ **Complete report** - LaTeX format with analysis  
✅ **Models included** - Ready for submission  

## 📝 Usage

The project demonstrates:
- From-scratch transformer implementation
- Minimal pairs evaluation methodology
- Model size impact analysis
- Resource-efficient training
- Professional report generation

## 🔧 Technical Details

- **Framework**: PyTorch 2.0+
- **Architecture**: Transformer with causal attention
- **Training**: Adam optimizer, 3 epochs
- **Evaluation**: 1000+ minimal pairs
- **Report**: LaTeX with tables and analysis

## 📄 License

This project is created for educational purposes as part of a university assignment. 
