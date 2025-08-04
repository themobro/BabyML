import re
from collections import Counter
import pickle

class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
        
    def _preprocess_text(self, text):
        """Simple text preprocessing"""
        # Convert to lowercase and tokenize by whitespace and punctuation
        text = text.lower()
        # Add spaces around punctuation
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        # Split by whitespace
        tokens = text.split()
        return tokens
    
    def train(self, texts):
        """Train tokenizer on a list of texts"""
        print("Training tokenizer...")
        
        # Collect all tokens
        all_tokens = []
        for text in texts:
            tokens = self._preprocess_text(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Build vocabulary (most frequent tokens)
        vocab_size_for_words = self.vocab_size - len(self.special_tokens)
        most_common = token_counts.most_common(vocab_size_for_words)
        
        # Initialize with special tokens
        self.word_to_id = self.special_tokens.copy()
        self.id_to_word = {v: k for k, v in self.special_tokens.items()}
        
        # Add most common words
        for i, (word, count) in enumerate(most_common):
            word_id = len(self.special_tokens) + i
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
        
        print(f"Tokenizer trained with vocabulary size: {len(self.word_to_id)}")
    
    def encode(self, text):
        """Encode text to token ids"""
        tokens = self._preprocess_text(text)
        token_ids = []
        
        for token in tokens:
            if token in self.word_to_id:
                token_ids.append(self.word_to_id[token])
            else:
                token_ids.append(self.word_to_id['<unk>'])
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token ids back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                tokens.append(self.id_to_word[token_id])
            else:
                tokens.append('<unk>')
        
        return ' '.join(tokens)
    
    def save(self, filepath):
        """Save tokenizer to file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, filepath):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word_to_id = data['word_to_id']
            self.id_to_word = data['id_to_word']
            self.vocab_size = data['vocab_size'] 