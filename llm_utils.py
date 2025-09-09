from transformers import pipeline
import logging
import time

HF_MODEL_DICTIONARY = {
    "GEMMA_SMALL": "google/gemma-3-1b-it",
    "GEMMA": "google/gemma-7b"
}

logger = logging.getLogger(__name__)

class LocalLlm:
    def __init__(self, model_name, system_prompt=""):
        self.model_name = model_name
        self.hf_model_name = HF_MODEL_DICTIONARY[model_name] if model_name in HF_MODEL_DICTIONARY else HF_MODEL_DICTIONARY["GEMMA_SMALL"]
        self.pipe = pipeline("text-generation", model=self.hf_model_name)
        self.messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt if len(system_prompt) > 0 else "You are a helpful assistant."}]}]
        logger.info("LLM created with model: %s and system prompt: %s", model_name , system_prompt)


    def get_response(self, text_message):
        start_time = time.time()
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text_message}]})
        response = self.pipe(self.messages, max_new_tokens=2000)
        logger.debug("Time taken: %.4f seconds and model response: %s", time.time() - start_time, response)
        response_text = response[0]["generated_text"][-1]['content']
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        return response_text
