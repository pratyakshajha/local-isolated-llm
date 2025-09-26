from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time
import torch

HF_MODEL_DICTIONARY = {
    "GEMMA_SMALL": "google/gemma-3-1b-it",
    "GEMMA": "google/gemma-3-4b-it",
    "CODE_GEMMA": "google/codegemma-1.1-2b",
    "GPT_2": "openai-community/gpt2",
    "LLAMA": "meta-llama/Llama-3.2-3B-Instruct",
    "LLAMA_SMALL": "meta-llama/Llama-3.2-1B-Instruct",
    "BERT": "google-bert/bert-base-uncased"
}

logger = logging.getLogger(__name__)

class LocalLlm:
    def __init__(self, model_name, system_prompt="You are a helpful assistant."):
        self.model_name = model_name
        self.hf_model_name = HF_MODEL_DICTIONARY[model_name] if model_name in HF_MODEL_DICTIONARY else HF_MODEL_DICTIONARY["GEMMA_SMALL"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_model_name, dtype=torch.bfloat16).to(self.device)
        self.messages = [{"role": "system", "content": system_prompt}]
        logger.info("LLM created with model: %s and system prompt: %s", model_name, system_prompt)

    def get_response(self, text_message):
        start_time = time.time()
        self.messages.append({"role": "user", "content": text_message})
        tokenized_chat = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        response = self.model.generate(tokenized_chat, max_new_tokens=2048)
        logger.debug("Time taken: %.4f seconds and model response: %s", time.time() - start_time, response)
        response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)[len(self.messages):]
        self.messages.append({"role": "assistant", "content": response_text})
        return response_text
