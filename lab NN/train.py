import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import create_small_model, create_large_model
from tokenizer import SimpleTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for text in texts:
            # Encode text
            token_ids = tokenizer.encode(text)
            
            # If text is shorter than max_length, pad it
            if len(token_ids) < max_length:
                # Pad with <pad> token (id=0)
                token_ids = token_ids + [0] * (max_length - len(token_ids))
                self.samples.append(token_ids)
            else:
                # Create sliding window samples for longer texts
                for i in range(0, len(token_ids) - max_length + 1, max_length // 2):
                    sample = token_ids[i:i + max_length]
                    if len(sample) == max_length:
                        self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample, dtype=torch.long)

def prepare_sample_data():
    """Prepare a small sample dataset for training"""
    # Simple sample texts (you could replace this with real data)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing involves understanding human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers are a type of neural network architecture.",
        "Language models predict the next word in a sequence.",
        "Training requires large amounts of text data.",
        "Attention mechanisms help models focus on relevant information.",
        "The cat sat on the mat and looked around.",
        "Python is a popular programming language for machine learning.",
        "Data preprocessing is an important step in model training.",
        "Neural networks consist of layers of interconnected nodes.",
        "Backpropagation is used to train neural networks.",
        "Gradient descent optimizes model parameters.",
        "Overfitting occurs when models memorize training data.",
        "Regularization techniques help prevent overfitting.",
        "Cross-validation helps evaluate model performance.",
        "Feature engineering can improve model accuracy.",
        "Ensemble methods combine multiple models for better predictions.",
        "Transfer learning uses pre-trained models for new tasks."
    ]
    
    # Repeat texts to create more training data
    extended_texts = sample_texts * 10
    
    return extended_texts

def train_model(model, dataloader, optimizer, device, epochs=5):
    """Train a model"""
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch, labels=batch)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
    
    return losses

def save_model_and_config(model, tokenizer, config, save_path):
    """Save model, tokenizer, and configuration"""
    os.makedirs(save_path, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pt'))
    
    # Save tokenizer
    tokenizer.save(os.path.join(save_path, 'tokenizer.pkl'))
    
    # Save config
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Prepare data
    print("Preparing sample data...")
    texts = prepare_sample_data()
    
    # Train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=500)
    tokenizer.train(texts)
    
    # Create datasets
    dataset = TextDataset(texts, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Vocabulary size: {len(tokenizer.word_to_id)}")
    
    # Configuration for both models
    small_config = {
        'model_type': 'small',
        'd_model': 64,
        'n_heads': 2,
        'n_layers': 2,
        'd_ff': 256,
        'vocab_size': len(tokenizer.word_to_id)
    }
    
    large_config = {
        'model_type': 'large',
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 4,
        'd_ff': 512,
        'vocab_size': len(tokenizer.word_to_id)
    }
    
    # Train small model
    print("\n" + "="*50)
    print("Training Small BabyLM")
    print("="*50)
    
    small_model = create_small_model(len(tokenizer.word_to_id)).to(device)
    small_optimizer = torch.optim.Adam(small_model.parameters(), lr=0.001)
    
    small_losses = train_model(small_model, dataloader, small_optimizer, device, epochs=3)
    save_model_and_config(small_model, tokenizer, small_config, 'models/small_babylm')
    
    # Train large model
    print("\n" + "="*50)
    print("Training Large BabyLM")
    print("="*50)
    
    large_model = create_large_model(len(tokenizer.word_to_id)).to(device)
    large_optimizer = torch.optim.Adam(large_model.parameters(), lr=0.001)
    
    large_losses = train_model(large_model, dataloader, large_optimizer, device, epochs=3)
    save_model_and_config(large_model, tokenizer, large_config, 'models/large_babylm')
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(small_losses, label='Small Model', marker='o')
    plt.plot(large_losses, label='Large Model', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()
    
    print("\nTraining completed!")
    print("Models saved in 'models/' directory")
    print("Training curves saved as 'training_curves.png'")

if __name__ == "__main__":
    main() 