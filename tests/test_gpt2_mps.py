import torch
import unittest
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class TestGPT2MPS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"\nUsing device: {cls.device}")
        
        # Load model and tokenizer
        print("Loading GPT-2 model and tokenizer...")
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Set pad token to eos token to enable padding
        cls.tokenizer.pad_token = cls.tokenizer.eos_token
        cls.model = GPT2LMHeadModel.from_pretrained('gpt2').to(cls.device)
        cls.model.eval()  # Set to evaluation mode

    def test_model_generation(self):
        # Test prompt
        prompt = "The quick brown fox"
        
        # Tokenize input and create attention mask
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nInput prompt: {prompt}")
        print(f"Generated text: {generated_text}")
        
        # Basic assertions
        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), len(prompt))
        self.assertTrue(generated_text.startswith(prompt))

    def test_model_inference(self):
        # Test batch processing
        prompts = [
            "Hello, how are",
            "The weather is",
            "I love to"
        ]
        
        # Tokenize inputs with padding
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=20,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
        # Check output shapes
        self.assertEqual(outputs.logits.device.type, self.device.type)
        self.assertEqual(outputs.logits.shape[0], len(prompts))
        self.assertEqual(outputs.logits.shape[2], self.model.config.vocab_size)

if __name__ == '__main__':
    unittest.main() 