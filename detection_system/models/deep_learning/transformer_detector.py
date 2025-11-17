"""
Deep Learning Transformer Model for Prompt Hacking Detection
Author: AI Security Team
Date: November 2024

Lý do: Sử dụng deep learning để capture complex patterns mà traditional ML bỏ lỡ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    DistilBertModel, DistilBertTokenizer,
    get_linear_schedule_with_warmup
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle
import json
from pathlib import Path
import time
from tqdm import tqdm

class PromptHackingDataset(Dataset):
    """
    Custom Dataset cho prompt hacking detection
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerPromptDetector(nn.Module):
    """
    Transformer-based model cho prompt hacking detection
    Architecture: DistilBERT + Classification Head + Dropout
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout_rate=0.3):
        super(TransformerPromptDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer
        self.transformer = DistilBertModel.from_pretrained(model_name)
        
        # Classification head
        hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits

class DeepLearningTrainer:
    """
    Trainer cho deep learning model
    """
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Model will be initialized during training
        self.model = None
        self.best_model_state = None
        
    def prepare_data(self, train_df, test_df=None, test_size=0.2):
        """
        Prepare data cho training
        """
        print("📊 Preparing training data...")
        
        # Convert labels to binary if needed
        if 'label' in train_df.columns:
            # Handle string labels
            label_mapping = {'benign': 0, 'malicious': 1}
            if train_df['label'].dtype == 'object':
                train_df['binary_label'] = train_df['label'].map(label_mapping)
            else:
                train_df['binary_label'] = train_df['label']
        else:
            raise ValueError("Dataset must have 'label' column")
        
        # Split training data if no test set provided
        if test_df is None:
            X_train, X_val, y_train, y_val = train_test_split(
                train_df['prompt'].values,
                train_df['binary_label'].values,
                test_size=test_size,
                random_state=42,
                stratify=train_df['binary_label'].values
            )
        else:
            X_train = train_df['prompt'].values
            y_train = train_df['binary_label'].values
            
            # Handle test set labels
            if test_df['label'].dtype == 'object':
                test_df['binary_label'] = test_df['label'].map(label_mapping)
            else:
                test_df['binary_label'] = test_df['label']
                
            X_val = test_df['prompt'].values
            y_val = test_df['binary_label'].values
        
        # Create datasets
        train_dataset = PromptHackingDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = PromptHackingDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
        # Class distribution
        train_labels = pd.Series(y_train)
        print(f"📊 Training class distribution:")
        print(f"   Benign: {sum(train_labels == 0)} ({sum(train_labels == 0)/len(train_labels)*100:.1f}%)")
        print(f"   Malicious: {sum(train_labels == 1)} ({sum(train_labels == 1)/len(train_labels)*100:.1f}%)")
        
        return train_dataset, val_dataset, X_val, y_val
    
    def train(self, train_dataset, val_dataset, X_val, y_val, 
              batch_size=16, epochs=5, learning_rate=2e-5, weight_decay=0.01):
        """
        Train deep learning model
        """
        print(f"🚀 Starting deep learning training...")
        print(f"📋 Config: batch_size={batch_size}, epochs={epochs}, lr={learning_rate}")
        
        # Initialize model
        self.model = TransformerPromptDetector(self.model_name)
        self.model.to(self.device)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,  # 10% warmup
            num_training_steps=total_steps
        )
        
        # Loss function với class weighting for imbalanced data
        class_counts = np.bincount(train_dataset.labels)
        class_weights = torch.FloatTensor([1.0, class_counts[0] / class_counts[1]]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training loop
        best_f1 = 0
        training_history = []
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            total_loss = 0
            train_predictions = []
            train_labels = []
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                # Store predictions
                predictions = torch.argmax(outputs, dim=-1)
                train_predictions.extend(predictions.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            train_f1 = f1_score(train_labels, train_predictions)
            
            # Validation phase
            val_f1, val_loss, val_predictions = self.evaluate(val_loader, criterion)
            
            print(f"📊 Results:")
            print(f"   Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
            print(f"   Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                self.best_model_state = self.model.state_dict().copy()
                print(f"   ✅ New best F1: {best_f1:.4f}")
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_f1': val_f1
            })
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation
        print(f"\n🎯 FINAL EVALUATION (Best F1: {best_f1:.4f})")
        final_predictions = self.predict(X_val)
        final_report = classification_report(y_val, final_predictions, 
                                           target_names=['benign', 'malicious'], 
                                           digits=4)
        print(final_report)
        
        return training_history, best_f1
    
    def evaluate(self, val_loader, criterion):
        """
        Evaluate model on validation set
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                batch_predictions = torch.argmax(outputs, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        f1 = f1_score(labels, predictions)
        
        return f1, avg_loss, predictions
    
    def predict(self, texts):
        """
        Predict on new texts
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        predictions = []
        
        # Create temporary dataset
        dummy_labels = [0] * len(texts)  # Dummy labels
        temp_dataset = PromptHackingDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch in temp_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                batch_predictions = torch.argmax(outputs, dim=-1)
                predictions.extend(batch_predictions.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, texts):
        """
        Predict probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        probabilities = []
        
        # Create temporary dataset
        dummy_labels = [0] * len(texts)  # Dummy labels
        temp_dataset = PromptHackingDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        temp_loader = DataLoader(temp_dataset, batch_size=16, shuffle=False)
        
        with torch.no_grad():
            for batch in temp_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probs = F.softmax(outputs, dim=-1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def save_model(self, save_path):
        """
        Save trained model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.best_model_state or self.model.state_dict(), 
                  save_path / 'model.pth')
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path / 'tokenizer')
        
        # Save config
        config = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'num_classes': 2
        }
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Model saved to {save_path}")
    
    def load_model(self, model_path):
        """
        Load trained model
        """
        model_path = Path(model_path)
        
        # Load config
        with open(model_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize model
        self.model = TransformerPromptDetector(config['model_name'])
        self.model.load_state_dict(torch.load(model_path / 'model.pth', map_location=self.device))
        self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path / 'tokenizer')
        self.max_length = config['max_length']
        
        print(f"✅ Model loaded from {model_path}")

def main():
    """
    Main training function
    """
    print("🤖 DEEP LEARNING PROMPT HACKING DETECTOR")
    print("=" * 60)
    
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    datasets_dir = project_root / "datasets"
    models_dir = project_root / "detection_system" / "saved_models" / "deep_learning"
    
    # Load datasets
    print("📂 Loading datasets...")
    
    # Try to load HuggingFace dataset first (larger)
    train_df = None
    test_df = None
    
    hf_files = list(datasets_dir.glob("huggingface_train_*.csv"))
    if hf_files:
        train_file = hf_files[0]
        test_file = datasets_dir / train_file.name.replace("_train_", "_test_")
        
        if test_file.exists():
            print(f"📊 Trying HuggingFace dataset: {train_file.name}")
            try:
                train_df = pd.read_csv(train_file, engine='python')
                test_df = pd.read_csv(test_file, engine='python')
                print(f"✅ Successfully loaded HuggingFace dataset!")
            except Exception as e:
                print(f"❌ Failed to load HuggingFace dataset: {e}")
                print("🔄 Falling back to challenging dataset...")
                train_df = None
                test_df = None
        else:
            print(f"📊 Trying HuggingFace dataset (train only): {train_file.name}")
            try:
                train_df = pd.read_csv(train_file, engine='python')
                test_df = None
                print(f"✅ Successfully loaded HuggingFace dataset!")
            except Exception as e:
                print(f"❌ Failed to load HuggingFace dataset: {e}")
                print("🔄 Falling back to challenging dataset...")
                train_df = None
                test_df = None
    
    # Fallback to challenging dataset if HuggingFace failed
    if train_df is None:
        challenging_files = list(datasets_dir.glob("challenging_train_*.csv"))
        if challenging_files:
            train_file = challenging_files[0]
            test_file = datasets_dir / train_file.name.replace("_train_", "_test_")
            
            print(f"📊 Using Challenging dataset: {train_file.name}")
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file) if test_file.exists() else None
        else:
            raise FileNotFoundError("No training datasets found!")
    
    print(f"✅ Loaded {len(train_df)} training samples")
    if test_df is not None:
        print(f"✅ Loaded {len(test_df)} test samples")
    
    # Initialize trainer
    trainer = DeepLearningTrainer()
    
    # Prepare data
    train_dataset, val_dataset, X_val, y_val = trainer.prepare_data(train_df, test_df)
    
    # Train model
    history, best_f1 = trainer.train(
        train_dataset, val_dataset, X_val, y_val,
        batch_size=16,
        epochs=5,
        learning_rate=2e-5
    )
    
    # Save model
    trainer.save_model(models_dir)
    
    print(f"\n🎯 TRAINING COMPLETED!")
    print(f"📈 Best F1 Score: {best_f1:.4f}")
    print(f"💾 Model saved to: {models_dir}")
    
    return trainer, best_f1

if __name__ == "__main__":
    trainer, best_f1 = main()
