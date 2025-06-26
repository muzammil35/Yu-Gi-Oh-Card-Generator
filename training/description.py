import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os

class YugiohDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def main():
    # Load your data
    print("Loading data...")
    try:
        with open('cards_data.json', 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    except:
        # If cards.json doesn't exist, try reading CSV
        df = pd.read_csv('cards.csv')
    
    # Select columns and format
    yugioh_df = df[['name', 'desc']].copy()
    print(f"Loaded {len(yugioh_df)} cards")
    
    # Format texts
    texts = []
    for _, row in yugioh_df.iterrows():
        text = f"Card Name: {row['name']}\nDescription: {row['desc']}<|endoftext|>"
        texts.append(text)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader
    dataset = YugiohDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{3}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    # Create output directory if it doesn't exist
    output_dir = "./yugioh-gpt2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    print("Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model training completed and saved to {output_dir}!")
    
    # Verify files were saved
    saved_files = os.listdir(output_dir)
    print(f"Saved files: {saved_files}")

if __name__ == "__main__":
    main()