# BabyLM Training and Evaluation Project

This project implements and compares two different BabyLM (Baby Language Model) architectures to investigate the relationship between model capacity and grammatical competence.

## Project Overview

Two transformer-based language models were implemented with distinct architectural differences to evaluate their performance on grammatical minimal pairs. The study demonstrates how model size affects linguistic understanding in small-scale language models.

## Model Architectures

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

## Implementation Details

### Training Configuration
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch size**: 8
- **Training epochs**: 3
- **Sequence length**: 64 tokens
- **Vocabulary size**: 112 tokens
- **Training samples**: 200

### Evaluation Methodology
Models were evaluated on 1,000 minimal pairs covering:
- Subject-verb agreement
- Auxiliary verb agreement
- Word order preferences
- Determiner-noun agreement

Each model was scored based on whether it assigned lower perplexity to grammatically correct sentences.

## Results

| Model | Accuracy | Performance |
|-------|----------|-------------|
| Small BabyLM | 88.3% | Baseline performance |
| Large BabyLM | 98.7% | Enhanced performance |
| **Improvement** | **+10.4 pp** | Significant benefit of increased capacity |

## Key Findings

1. **Model Size Impact**: The larger model demonstrates a 10.4 percentage point improvement in grammatical accuracy
2. **Resource Efficiency**: Both models train successfully on limited computational resources
3. **Grammatical Learning**: Both architectures learn meaningful linguistic patterns
4. **Scalable Performance**: Clear benefits from increased model capacity

## Project Structure

```
├── model.py              # Transformer implementation
├── tokenizer.py          # Simple tokenizer
├── train.py              # Training pipeline
├── evaluate.py           # Evaluation script

├── requirements.txt      # Dependencies
├── babylm_report.tex    # LaTeX report
├── training_curves.png   # Training visualization
├── model_comparison.png  # Results comparison
├── evaluation_results.json # Detailed results
└── models/               # Trained models
    ├── small_babylm/     # Small model checkpoint
    └── large_babylm/     # Large model checkpoint
```

## Technical Implementation

- **Framework**: PyTorch 2.0+
- **Architecture**: Transformer with causal attention
- **Training**: Adam optimizer, 3 epochs
- **Evaluation**: 1,000+ minimal pairs
- **Report**: LaTeX with comprehensive analysis

## Conclusion

This study demonstrates that even small transformer models can learn meaningful grammatical patterns, with larger models showing clear performance improvements on minimal pairs evaluation. The results support the hypothesis that increased model capacity leads to better linguistic understanding in language models.

## Files Description

- **`model.py`**: Complete transformer implementation
- **`train.py`**: Training pipeline with data generation
- **`evaluate.py`**: Minimal pairs evaluation

- **`babylm_report.tex`**: Academic report
- **`models/`**: Trained model checkpoints with configurations

## License

This project is created for educational purposes as part of a university assignment. 
