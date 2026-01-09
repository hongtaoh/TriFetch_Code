import torch
import torch.nn.functional as F
import re
import time
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ModelInterface:
    def __init__(self, model_key=None, config_path="config.yaml"):
        """
        Initialize model from config.
        
        Args:
            model_key: Key from config (e.g., "smollm-135m"). 
                      If None, uses default from config.
        """
        config = load_config(config_path)
        
        # Use default if not specified
        if model_key is None:
            model_key = config["default"]
        
        model_config = config["models"][model_key]
        self.model_name = model_config["name"]
        self.description = model_config.get("description", self.model_name)
        
        # Device setup
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Loading {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.reference_model = self.model
        
        print(f"✅ Model loaded!")

    def generate_text(self, prompt, max_new_tokens=256):
        messages = [{"role": "user", "content": prompt}]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated

    def get_sequence_log_prob(self, prompt, response, use_reference=False):
        model = self.reference_model if use_reference else self.model
        
        full_text = prompt + response
        
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        prompt_length = prompt_tokens['input_ids'].shape[1]
        
        with torch.no_grad():
            outputs = model(**full_tokens)
            logits = outputs.logits
        
        log_probs = F.log_softmax(logits, dim=-1)
        input_ids = full_tokens['input_ids']
        
        token_log_probs = log_probs[:, :-1, :].gather(
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        response_log_probs = token_log_probs[:, prompt_length-1:]
        total_log_prob = response_log_probs.sum()
        
        return total_log_prob

    def generate_verified_traces(self, question, answer_key):
        valid_traces = []
        max_honest_attempts = 3
        
        yield "log", f"--- 🎬 STARTING SAMPLING ---"
        yield "log", f"   Model: {self.model_name}"

        for i in range(max_honest_attempts):
            if len(valid_traces) >= 3:
                break

            try:
                prompt = f"""{question}

    Think step by step, then give your answer.

    ANSWER:"""

                response = self.generate_text(prompt, max_new_tokens=250)
                time.sleep(2)
                
                # Extract answer - multiple strategies
                guessed = self._extract_answer(response)
                
                if guessed:
                    full_response = f"{response}\n\nANSWER: {guessed}"
                    
                    if guessed == answer_key:
                        yield "success", f"✅ Attempt {i+1}: Correct ({guessed})"
                        valid_traces.append(full_response)
                    else:
                        yield "error", f"❌ Attempt {i+1}: Wrong ({guessed}, expected {answer_key})"
                else:
                    # This should rarely happen now
                    yield "warning", f"⚠️ Attempt {i+1}: No answer found"
                        
            except Exception as e:
                yield "error", f"Error: {e}"

        # PHASE 2: Fallback
        needed = 3 - len(valid_traces)
        if needed > 0:
            yield "warning", f"--- Fallback needed for {needed} traces ---"
            
            for i in range(needed):
                fallback_prompt = f"""{question}

                    The correct answer is {answer_key}. Explain why {answer_key} is correct.

                    """

                try:
                    response = self.generate_text(fallback_prompt, max_new_tokens=300)
                    time.sleep(2)
                    
                    full_response = f"{response}\n\nANSWER: {answer_key}"
                    valid_traces.append(full_response)
                    yield "success", f"✅ Fallback {i+1} done"
                    
                except Exception as e:
                    yield "error", f"Error: {e}"

        yield "log", "--- 🎉 Complete ---"
        yield "result", valid_traces


    def _extract_answer(self, response):
        """
        Extract answer from response. Tries multiple strategies.
        Returns A, B, C, or D, or None if nothing found.
        """
        text = response.upper()
        
        # Strategy 1: Look for "ANSWER: X" or "ANSWER X"
        match = re.search(r'ANSWER[:\s]+([A-D])', text)
        if match:
            return match.group(1)
        
        # Strategy 2: Look for "X)" or "X." at start of line (like "A) Influenza")
        match = re.search(r'\b([A-D])\s*[\)\.:]', text)
        if match:
            return match.group(1)
        
        # Strategy 3: Look for "is X" or "is option X"
        match = re.search(r'IS\s+(?:OPTION\s+)?([A-D])', text)
        if match:
            return match.group(1)
        
        # Strategy 4: Look for "choose X" or "select X"
        match = re.search(r'(?:CHOOSE|SELECT)\s+([A-D])', text)
        if match:
            return match.group(1)
        
        # Strategy 5: Find ALL standalone A/B/C/D, take the LAST one
        matches = re.findall(r'\b([A-D])\b', text)
        if matches:
            return matches[-1]
        
        # Strategy 6: Find any A, B, C, D anywhere
        for letter in ['A', 'B', 'C', 'D']:
            if letter in text:
                return letter
        
        return None


def get_available_models(config_path="config.yaml"):
    """Get models for UI dropdown."""
    config = load_config(config_path)
    return {
        key: cfg.get("description", cfg["name"])
        for key, cfg in config["models"].items()
    }