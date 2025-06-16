# Complete Small Language Model Training Pipeline
# For Academic Lecture Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import re
import os
import glob
from collections import Counter
import math
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

# PDF processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("Warning: Neither PyPDF2 nor pdfplumber found. Install one for PDF support:")
        print("pip install PyPDF2  OR  pip install pdfplumber")

# =======================
# 1. DATA PREPROCESSING
# =======================

class TextPreprocessor:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using available PDF library"""
        text = ""

        try:
            # Try pdfplumber first (generally better text extraction)
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                pass

            # Fallback to PyPDF2
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                pass

        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""

        print(f"No PDF library available to process {pdf_path}")
        return ""

    def load_folder_text(self, folder_path, file_extensions=None):
        """Load and clean text from all files in a folder"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.text', '.pdf']

        all_text = ""
        files_processed = 0

        print(f"Loading files from folder: {folder_path}")

        # Get all files with specified extensions
        all_files = []
        for ext in file_extensions:
            pattern = os.path.join(folder_path, f"**/*{ext}")
            all_files.extend(glob.glob(pattern, recursive=True))

        if not all_files:
            print(f"No files found with extensions {file_extensions} in {folder_path}")
            return ""

        for file_path in sorted(all_files):
            try:
                print(f"Processing: {os.path.basename(file_path)}")

                # Handle PDF files differently
                if file_path.lower().endswith('.pdf'):
                    if not PDF_AVAILABLE:
                        print(f"Skipping PDF {file_path}: No PDF library installed")
                        continue
                    text = self.extract_text_from_pdf(file_path)
                else:
                    # Handle text files
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                if text:
                    # Clean text
                    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
                    text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)  # Keep basic punctuation
                    text = text.strip()

                    if text:  # Only add non-empty text
                        all_text += f"\n\n{text}"
                        files_processed += 1
                        print(f"  ✓ Extracted {len(text)} characters")
                else:
                    print(f"  ⚠ No text extracted from {os.path.basename(file_path)}")

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        print(f"Successfully processed {files_processed} files")
        print(f"Total text length: {len(all_text)} characters")

        return all_text.strip()

    def load_and_clean_text(self, file_paths):
        """Load and clean text from multiple files (backward compatibility)"""
        all_text = ""

        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # Clean text
                text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
                text = re.sub(r'[^\w\s.,!?;:\-\'"()]', '', text)  # Keep basic punctuation
                all_text += text + " "

        return all_text.strip()

    def build_vocabulary(self, text, min_freq=2):
        """Build vocabulary from text"""
        # Tokenize (simple word-level tokenization)
        words = text.lower().split()
        word_counts = Counter(words)

        # Filter by minimum frequency
        filtered_words = {word: count for word, count in word_counts.items()
                         if count >= min_freq}

        # Build vocab
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

        for i, word in enumerate(sorted(filtered_words.keys())):
            self.vocab[word] = i + 4

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        print(f"Vocabulary size: {self.vocab_size}")
        return self.vocab

    def text_to_sequence(self, text):
        """Convert text to sequence of token IDs"""
        words = text.lower().split()
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

    def sequence_to_text(self, sequence):
        """Convert sequence of token IDs back to text"""
        return ' '.join([self.reverse_vocab.get(idx, '<UNK>') for idx in sequence])

# =======================
# 2. DATASET CLASS
# =======================

class LectureDataset(Dataset):
    def __init__(self, text_sequences, seq_length=128):
        self.sequences = text_sequences
        self.seq_length = seq_length

    def __len__(self):
        return len(self.sequences) - self.seq_length

    def __getitem__(self, idx):
        # Input sequence and target (shifted by one)
        input_seq = torch.tensor(self.sequences[idx:idx + self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.sequences[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return input_seq, target_seq

# =======================
# 3. TRANSFORMER MODEL
# =======================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)

        output = self.W_o(attention_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len):
        """Create causal mask for decoder"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb

        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(x.device)

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

# =======================
# 4. TRAINING FUNCTIONS
# =======================

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, (input_seq, target_seq) in enumerate(tqdm(self.train_loader, desc="Training")):
            input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(input_seq)

            # Calculate loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for input_seq, target_seq in tqdm(self.val_loader, desc="Validating"):
                input_seq, target_seq = input_seq.to(self.device), target_seq.to(self.device)

                logits = self.model(input_seq)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq.view(-1))
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

    def save_checkpoint(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

# =======================
# 5. TEXT GENERATION
# =======================

class TextGenerator:
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device

    def generate_text(self, prompt, max_length=100, temperature=1.0, top_k=50):
        self.model.eval()

        # Encode prompt
        tokens = [self.preprocessor.vocab['<SOS>']] + self.preprocessor.text_to_sequence(prompt)

        with torch.no_grad():
            for _ in range(max_length):
                # Convert to tensor
                input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

                # Get model predictions
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Check for end token or unknown token
                if next_token == self.preprocessor.vocab.get('<EOS>', -1):
                    break

                tokens.append(next_token)

        # Decode generated tokens
        generated_text = self.preprocessor.sequence_to_text(tokens[1:])  # Skip <SOS>
        return generated_text
# =======================
# 6. MAIN TRAINING SCRIPT
# =======================

def main():
    # Configuration
    CONFIG = {
        'folder_path': './all lecture',  # Path to your folder containing lecture files
        'file_extensions': ['.txt', '.md', '.text', '.pdf'],  # File types to process
        'seq_length': 128,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 20,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Using device: {CONFIG['device']}")

    # 1. Preprocess data
    print("Loading and preprocessing data...")
    preprocessor = TextPreprocessor()

    # Check if folder exists
    if os.path.exists(CONFIG['folder_path']):
        print(f"Found folder: {CONFIG['folder_path']}")
        text = preprocessor.load_folder_text(
            CONFIG['folder_path'],
            CONFIG['file_extensions']
        )

    # Build vocabulary
    vocab = preprocessor.build_vocabulary(text, min_freq=2)

    # Convert text to sequences
    sequences = preprocessor.text_to_sequence(text)

    # Check if we have enough data
    if len(sequences) < CONFIG['seq_length'] * 2:
        print("Warning: Very limited data. Consider adding more text files or reducing seq_length.")

    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(sequences))
    train_sequences = sequences[:split_idx]
    val_sequences = sequences[split_idx:]

    # Create datasets
    train_dataset = LectureDataset(train_sequences, CONFIG['seq_length'])
    val_dataset = LectureDataset(val_sequences, CONFIG['seq_length'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 2. Initialize model
    model = SmallLanguageModel(
        vocab_size=preprocessor.vocab_size,
        d_model=CONFIG['d_model'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        d_ff=CONFIG['d_ff']
    ).to(CONFIG['device'])

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])

    # 4. Train model
    trainer = Trainer(model, train_loader, val_loader, optimizer, CONFIG['device'])
    trainer.train(CONFIG['epochs'])

    # 5. Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': preprocessor.vocab,
        'reverse_vocab': preprocessor.reverse_vocab,
        'config': CONFIG
    }, 'final_slm_model.pt')

    # Save preprocessor
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)

    print("Training completed! Model saved as 'final_slm_model.pt'")

    # 6. Test text generation
    print("\n" + "="*50)
    print("TESTING TEXT GENERATION")
    print("="*50)

    generator = TextGenerator(model, preprocessor, CONFIG['device'])

    test_prompts = [
        "what is multivariate analysis",
        "what is PCA",
        "what is CCA"
    ]

    for prompt in test_prompts:
        generated = generator.generate_text(prompt, max_length=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")

# =======================
# 7. INFERENCE SCRIPT
# =======================

def load_and_generate():
    """Load trained model and generate text"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load preprocessor
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)

    # Load model
    checkpoint = torch.load('final_slm_model.pt', map_location=device)
    config = checkpoint['config']

    model = SmallLanguageModel(
        vocab_size=len(checkpoint['vocab']),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate text
    generator = TextGenerator(model, preprocessor, device)

    while True:
        prompt = input("\nEnter a prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break

        generated = generator.generate_text(prompt, max_length=100, temperature=0.8)
        print(f"Generated: {generated}")

if __name__ == "__main__":
    main()

    # Uncomment the line below to run inference after training
    # load_and_generate()