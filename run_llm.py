from logging_config import setup_logging
from llm_utils import LocalLlm

setup_logging()

# GEMMA_SMALL,GEMMA,CODE_GEMMA,LLAMA,LLAMA_SMALL,BERT
local_llm = LocalLlm("GEMMA_SMALL")

print(f"{local_llm.model_name} is ready! (Type 'exit' to quit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    response = local_llm.get_response(user_input)
    print(f"{local_llm.model_name}: {response}")