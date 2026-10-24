"""Dependencies shared by API route handlers."""

from starlette.requests import HTTPConnection

from app.services.chat_service import ChatService
from app.services.model_service import ModelService


def get_chat_service(connection: HTTPConnection) -> ChatService:
    return connection.app.state.chat_service


def get_model_service(connection: HTTPConnection) -> ModelService:
    return connection.app.state.model_service
