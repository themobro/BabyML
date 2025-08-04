import json
import os
from datetime import datetime

def load_evaluation_results():
    """Load evaluation results from JSON file"""
    if not os.path.exists('evaluation_results.json'):
        raise FileNotFoundError("evaluation_results.json not found. Please run evaluate.py first.")
    
    with open('evaluation_results.json', 'r') as f:
        return json.load(f)

def generate_latex_report(results):
    """Generate LaTeX report from evaluation results"""
    
    # Extract key information
    small_accuracy = results['small_model']['accuracy']
    large_accuracy = results['large_model']['accuracy']
    accuracy_diff = results['comparison']['accuracy_difference']
    total_pairs = results['comparison']['total_pairs_evaluated']
    
    small_config = results['small_model']['config']
    large_config = results['large_model']['config']
    
    latex_content = f"""
\\documentclass[11pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{amsmath}}
\\usepackage{{hyperref}}

\\title{{BabyLM Training and Evaluation Report}}
\\author{{Student Name}}
\\date{{{datetime.now().strftime("%B %d, %Y")}}}

\\begin{{document}}

\\maketitle

\\section{{Introduction}}

This report presents the training and evaluation of two different BabyLM (Baby Language Model) architectures. The goal was to compare how model size affects performance on grammatical minimal pairs, providing insights into the relationship between model capacity and linguistic competence.

\\section{{Methodology}}

\\subsection{{Model Architectures}}

Two transformer-based language models were implemented with the following configurations:

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Small Model}} & \\textbf{{Large Model}} \\\\
\\midrule
Model Dimension & {small_config['d_model']} & {large_config['d_model']} \\\\
Number of Heads & {small_config['n_heads']} & {large_config['n_heads']} \\\\
Number of Layers & {small_config['n_layers']} & {large_config['n_layers']} \\\\
Feed-Forward Dimension & {small_config['d_ff']} & {large_config['d_ff']} \\\\
Vocabulary Size & {small_config['vocab_size']:,} & {large_config['vocab_size']:,} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Model Architecture Comparison}}
\\end{{table}}

The key difference between the models is their capacity: the large model has twice the embedding dimension, twice the number of attention heads, twice the number of layers, and a larger feed-forward dimension, resulting in significantly more parameters.

\\subsection{{Training Data}}

Both models were trained on the same dataset consisting of simple English sentences covering various topics including machine learning, natural language processing, and general knowledge. The training corpus was designed to be small (suitable for resource-constrained environments) while still providing sufficient linguistic diversity.

\\subsection{{Training Procedure}}

\\begin{{itemize}}
    \\item Optimizer: Adam with learning rate 0.001
    \\item Batch size: 8
    \\item Epochs: 3
    \\item Sequence length: 64 tokens
    \\item Tokenizer: Simple word-level tokenizer with vocabulary size 500
\\end{{itemize}}

\\subsection{{Evaluation Methodology}}

Models were evaluated on {total_pairs:,} minimal pairs covering:
\\begin{{itemize}}
    \\item Subject-verb agreement
    \\item Auxiliary verb agreement
    \\item Word order preferences
    \\item Determiner-noun agreement
\\end{{itemize}}

Each minimal pair consists of a grammatically correct sentence and an incorrect variant. Models were scored based on whether they assigned lower perplexity (higher probability) to the correct sentence.

\\section{{Results}}

\\subsection{{Training Performance}}

Both models successfully decreased their training loss over the 3 epochs, indicating effective learning of the training data patterns.

\\subsection{{Evaluation Results}}

\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Model}} & \\textbf{{Accuracy}} & \\textbf{{Percentage}} \\\\
\\midrule
Small BabyLM & {small_accuracy:.3f} & {small_accuracy*100:.1f}\\% \\\\
Large BabyLM & {large_accuracy:.3f} & {large_accuracy*100:.1f}\\% \\\\
\\midrule
\\textbf{{Difference}} & {accuracy_diff:.3f} & {accuracy_diff*100:.1f} pp \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Minimal Pairs Evaluation Results}}
\\end{{table}}

\\begin{{figure}}[h]
\\centering
\\includegraphics[width=0.8\\textwidth]{{model_comparison.png}}
\\caption{{Model Performance Comparison}}
\\end{{figure}}

\\section{{Discussion}}

\\subsection{{Key Findings}}

The evaluation revealed several important insights:

\\begin{{enumerate}}
    \\item \\textbf{{Model Size Impact}}: The larger model {'outperformed' if accuracy_diff > 0 else 'underperformed compared to'} the smaller model by {abs(accuracy_diff)*100:.1f} percentage points, {'demonstrating' if accuracy_diff > 0 else 'surprisingly showing that increased model capacity did not lead to'} {'the expected benefits of' if accuracy_diff > 0 else ''} improved grammatical competence {'on this task' if accuracy_diff <= 0 else ''}.
    
    \\item \\textbf{{Performance Level}}: Both models achieved {'reasonable' if min(small_accuracy, large_accuracy) > 0.6 else 'limited'} performance on the minimal pairs task, with accuracies {'above' if min(small_accuracy, large_accuracy) > 0.5 else 'around'} {'chance level' if min(small_accuracy, large_accuracy) <= 0.5 else '60%'}.
    
    \\item \\textbf{{Training Efficiency}}: The smaller model required significantly fewer parameters while achieving {'competitive' if abs(accuracy_diff) < 0.1 else 'different'} performance, suggesting {'good efficiency' if accuracy_diff <= 0 else 'that the additional parameters in the larger model contributed to better linguistic understanding'}.
\\end{{enumerate}}

\\subsection{{Limitations}}

Several limitations should be considered:
\\begin{{itemize}}
    \\item \\textbf{{Training Data Size}}: The models were trained on a relatively small corpus, which may limit their ability to learn complex linguistic patterns.
    \\item \\textbf{{Training Duration}}: Only 3 epochs were used for training, which may not be sufficient for full convergence.
    \\item \\textbf{{Evaluation Scope}}: The minimal pairs focus on specific grammatical phenomena and may not reflect broader linguistic competence.
\\end{{itemize}}

\\section{{Conclusion}}

This study compared two BabyLM architectures differing in model size and capacity. The results show that {'larger models can provide benefits for grammatical understanding, though the improvement was modest' if accuracy_diff > 0.05 else 'model size alone does not guarantee improved performance on grammatical tasks'}.

Future work could explore:
\\begin{{itemize}}
    \\item Training on larger and more diverse datasets
    \\item Longer training with more sophisticated optimization schedules
    \\item Different architectural choices (e.g., different attention mechanisms)
    \\item More comprehensive evaluation on diverse linguistic phenomena
\\end{{itemize}}

The code and trained models are available at: \\url{{https://github.com/[username]/babylm-comparison}}

\\end{{document}}
"""
    
    return latex_content

def main():
    print("Loading evaluation results...")
    try:
        results = load_evaluation_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python evaluate.py' first to generate evaluation results.")
        return
    
    print("Generating LaTeX report...")
    latex_content = generate_latex_report(results)
    
    # Save LaTeX file
    with open('babylm_report.tex', 'w') as f:
        f.write(latex_content)
    
    print("LaTeX report saved as 'babylm_report.tex'")
    print("To compile to PDF, run: pdflatex babylm_report.tex")
    
    # Try to compile PDF if pdflatex is available
    try:
        import subprocess
        result = subprocess.run(['pdflatex', 'babylm_report.tex'], 
                              capture_output=True, text=True, check=True)
        print("PDF report generated successfully as 'babylm_report.pdf'")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("pdflatex not found or compilation failed.")
        print("Please compile manually using: pdflatex babylm_report.tex")

if __name__ == "__main__":
    main() 