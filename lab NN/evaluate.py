import torch
import json
import os
import numpy as np
from model import BabyLM
from tokenizer import SimpleTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_model_and_tokenizer(model_path):
    """Load a trained model and its tokenizer"""
    # Load config
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(os.path.join(model_path, 'tokenizer.pkl'))
    
    # Create and load model
    max_len = config.get('max_len', 256 if config['model_type'] == 'small' else 512)
    model = BabyLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_len=max_len
    )
    
    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), map_location='cpu'))
    
    return model, tokenizer, config

def create_minimal_pairs():
    """Create sample minimal pairs for evaluation"""
    minimal_pairs = [
        # Subject-verb agreement
        ("The cat runs quickly.", "The cats run quickly."),
        ("She walks to school.", "She walk to school."),
        ("The dog barks loudly.", "The dogs bark loudly."),
        ("He eats breakfast.", "He eat breakfast."),
        ("The bird flies high.", "The birds fly high."),
        
        # Grammatical vs ungrammatical
        ("I am going home.", "I are going home."),
        ("They have finished work.", "They has finished work."),
        ("She is reading a book.", "She are reading a book."),
        ("We were at the park.", "We was at the park."),
        ("You are very smart.", "You is very smart."),
        
        # Word order
        ("The red beautiful car.", "The beautiful red car."),
        ("She quickly ran home.", "She ran quickly home."),
        ("They often visit museums.", "They visit often museums."),
        ("He always tells truth.", "He tells always truth."),
        ("We sometimes eat pizza.", "We eat sometimes pizza."),
        
        # Determiner agreement
        ("I have a apple.", "I have an apple."),
        ("She bought a book.", "She bought an book."),
        ("He needs a umbrella.", "He needs an umbrella."),
        ("They found a elephant.", "They found an elephant."),
        ("We saw a owl.", "We saw an owl."),
    ]
    
    # Extend with more pairs to reach 1000+ sentences
    extended_pairs = []
    templates = [
        ("The {noun} {verb}s well.", "The {noun}s {verb} well."),
        ("She {verb}s every day.", "She {verb} every day."),
        ("He is {adjective}.", "He are {adjective}."),
        ("They have {noun}.", "They has {noun}."),
        ("We {verb} {adverb}.", "We {verb}s {adverb}."),
    ]
    
    nouns = ["cat", "dog", "bird", "fish", "tree", "car", "house", "book", "phone", "computer"]
    verbs = ["run", "walk", "jump", "sleep", "eat", "work", "play", "study", "read", "write"]
    adjectives = ["happy", "sad", "big", "small", "fast", "slow", "good", "bad", "new", "old"]
    adverbs = ["quickly", "slowly", "carefully", "loudly", "quietly", "often", "sometimes", "always", "never", "well"]
    
    for template in templates:
        for noun in nouns:
            for verb in verbs:
                for adj in adjectives:
                    for adv in adverbs:
                        correct = template[0].format(noun=noun, verb=verb, adjective=adj, adverb=adv)
                        incorrect = template[1].format(noun=noun, verb=verb, adjective=adj, adverb=adv)
                        extended_pairs.append((correct, incorrect))
                        
                        if len(extended_pairs) >= 1000:  # Stop when we have enough pairs
                            break
                    if len(extended_pairs) >= 1000:
                        break
                if len(extended_pairs) >= 1000:
                    break
            if len(extended_pairs) >= 1000:
                break
        if len(extended_pairs) >= 1000:
            break
    
    # Combine original and extended pairs
    all_pairs = minimal_pairs + extended_pairs[:1000-len(minimal_pairs)]
    return all_pairs

def calculate_perplexity(model, tokenizer, text, device):
    """Calculate perplexity of a text given a model"""
    model.eval()
    
    # Tokenize text
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:  # Need at least 2 tokens for perplexity
        return float('inf')
    
    # Convert to tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss']
        perplexity = torch.exp(loss).item()
    
    return perplexity

def evaluate_on_minimal_pairs(model, tokenizer, minimal_pairs, device):
    """Evaluate model on minimal pairs"""
    model.eval()
    correct_predictions = 0
    total_pairs = len(minimal_pairs)
    
    results = []
    
    for correct_sentence, incorrect_sentence in tqdm(minimal_pairs, desc="Evaluating"):
        # Calculate perplexity for both sentences
        correct_ppl = calculate_perplexity(model, tokenizer, correct_sentence, device)
        incorrect_ppl = calculate_perplexity(model, tokenizer, incorrect_sentence, device)
        
        # Model should assign lower perplexity to correct sentence
        if correct_ppl < incorrect_ppl:
            correct_predictions += 1
            prediction = "correct"
        else:
            prediction = "incorrect"
        
        results.append({
            'correct_sentence': correct_sentence,
            'incorrect_sentence': incorrect_sentence,
            'correct_ppl': correct_ppl,
            'incorrect_ppl': incorrect_ppl,
            'prediction': prediction
        })
    
    accuracy = correct_predictions / total_pairs
    return accuracy, results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load models
    print("Loading models...")
    small_model, small_tokenizer, small_config = load_model_and_tokenizer('models/small_babylm')
    large_model, large_tokenizer, large_config = load_model_and_tokenizer('models/large_babylm')
    
    small_model.to(device)
    large_model.to(device)
    
    # Create minimal pairs
    print("Creating minimal pairs dataset...")
    minimal_pairs = create_minimal_pairs()
    print(f"Created {len(minimal_pairs)} minimal pairs")
    
    # Evaluate small model
    print("\nEvaluating Small BabyLM...")
    small_accuracy, small_results = evaluate_on_minimal_pairs(small_model, small_tokenizer, minimal_pairs, device)
    
    # Evaluate large model
    print("\nEvaluating Large BabyLM...")
    large_accuracy, large_results = evaluate_on_minimal_pairs(large_model, large_tokenizer, minimal_pairs, device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Small Model Accuracy: {small_accuracy:.3f} ({small_accuracy*100:.1f}%)")
    print(f"Large Model Accuracy: {large_accuracy:.3f} ({large_accuracy*100:.1f}%)")
    print(f"Difference: {(large_accuracy - small_accuracy)*100:.1f} percentage points")
    
    # Save detailed results
    evaluation_results = {
        'small_model': {
            'config': small_config,
            'accuracy': small_accuracy,
            'results': small_results[:10]  # Save first 10 examples
        },
        'large_model': {
            'config': large_config,
            'accuracy': large_accuracy,
            'results': large_results[:10]  # Save first 10 examples
        },
        'comparison': {
            'accuracy_difference': large_accuracy - small_accuracy,
            'total_pairs_evaluated': len(minimal_pairs)
        }
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    models = ['Small BabyLM', 'Large BabyLM']
    accuracies = [small_accuracy, large_accuracy]
    
    bars = plt.bar(models, accuracies, color=['lightblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('BabyLM Model Comparison on Minimal Pairs')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{accuracy:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some example predictions
    print("\nExample Predictions (first 5 pairs):")
    print("-" * 80)
    for i, (small_result, large_result) in enumerate(zip(small_results[:5], large_results[:5])):
        print(f"\nPair {i+1}:")
        print(f"Correct:   '{small_result['correct_sentence']}'")
        print(f"Incorrect: '{small_result['incorrect_sentence']}'")
        print(f"Small Model - Correct PPL: {small_result['correct_ppl']:.2f}, "
              f"Incorrect PPL: {small_result['incorrect_ppl']:.2f}, "
              f"Prediction: {small_result['prediction']}")
        print(f"Large Model - Correct PPL: {large_result['correct_ppl']:.2f}, "
              f"Incorrect PPL: {large_result['incorrect_ppl']:.2f}, "
              f"Prediction: {large_result['prediction']}")
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to 'evaluation_results.json'")
    print(f"Comparison plot saved to 'model_comparison.png'")

if __name__ == "__main__":
    main() 