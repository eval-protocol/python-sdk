from abc import ABC, abstractmethod
import logging
from fastapi import Request
from fastapi import HTTPException
import httpx
from typing import Optional

logger = logging.getLogger(__name__)


class AuthProvider(ABC):
    @abstractmethod
    def validate_and_return_account_id(self, request: Request) -> Optional[str]: ...


class NoAuthProvider(AuthProvider):
    def validate_and_return_account_id(self, request: Request) -> Optional[str]:
        return None
