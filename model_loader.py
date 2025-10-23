# model_loader.py
"""
Model Loader Module
-------------------
Loads and returns a Hugging Face conversational model and tokenizer wrapped in a pipeline.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch

set_seed(42)

class ModelLoader:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", use_gpu: bool = False):
        self.model_name = model_name
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        self.tokenizer = None
        self.model = None

    def load(self):
        """
        Loads the tokenizer, model, and creates the text generation pipeline.
        Returns: A transformers pipeline object.
        """
        try:
            print(f"Loading model: {self.model_name} on device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token 

            # Use the Hugging Face pipeline (Required)
            chatbot_pipeline = pipeline(
                "text-generation", 
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1, 
                config={"pad_token_id": self.tokenizer.eos_token_id} 
            )
            print("Model loaded successfully!\n")
            return chatbot_pipeline
            
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            return None