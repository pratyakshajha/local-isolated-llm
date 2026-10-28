"""Dependencies shared by API route handlers."""

from starlette.requests import HTTPConnection

from app.services.model_service import ModelService


def get_model_service(connection: HTTPConnection) -> ModelService:
    return connection.app.state.model_service
