from abc import ABC, abstractmethod
import logging
from fastapi import Request
from typing import Optional
from .models import AccountInfo

logger = logging.getLogger(__name__)


class AuthProvider(ABC):
    @abstractmethod
    def validate_and_return_account_info(self, request: Request) -> Optional[AccountInfo]: ...


class NoAuthProvider(AuthProvider):
    def validate_and_return_account_info(self, request: Request) -> Optional[AccountInfo]:
        return None
