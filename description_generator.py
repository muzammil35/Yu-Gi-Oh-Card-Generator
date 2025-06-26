import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from torchvision.utils import make_grid
from training.generator import Generator  # Assuming your Generator class is in generator.py
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF
import subprocess
import os
import shutil   
import threading  

class DescriptionGenerator:
    def __init__(self, model_dir="./fine_tuned_models/yugioh-gpt2"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        
        # Load models once during initialization
        self.load_models()
    
    def load_models(self):
        """Load both text generation and image generation models"""
        # Load text generation model
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory {self.model_dir} does not exist")
        
        print("Loading text generation model and tokenizer...")
        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        print("Text generation model loaded successfully!")
    
    def generate_description(self, card_name, max_length=150, temperature=0.8):
        """Generate description for a single card"""
        prompt = f"Card Name: {card_name}\nDescription:"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the description part
        if "Description:" in generated_text:
            description = generated_text.split("Description:", 1)[1].strip()
            return description
        return generated_text
 
    
    
    