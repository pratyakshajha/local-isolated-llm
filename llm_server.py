import asyncio
import websockets
from logging_config import setup_logging
from llm_utils import LocalLlm
import logging

setup_logging()

logger = logging.getLogger(__name__)

async def handler(websocket):
    # GEMMA_SMALL,GEMMA,CODE_GEMMA,LLAMA,LLAMA_SMALL,BERT
    model_name = "GEMMA_SMALL"
    logger.info(f"Starting the chat server with {model_name}")
    local_llm = LocalLlm(model_name)

    await websocket.send(f"{local_llm.model_name} is ready to chat!")
    
    while True:
        try:
            message = await websocket.recv()
            logger.info(f"Message received: {message}")
            response = local_llm.get_response(message)
            await websocket.send(f"{local_llm.model_name}: {response}")
        except websockets.ConnectionClosed:
            logger.info("Connection closed.")
            break

async def main():
    async with websockets.serve(handler, "localhost", 8900):
        await asyncio.Future()

if __name__ == "__main__":    
    asyncio.run(main())
