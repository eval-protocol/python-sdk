from abc import ABC, abstractmethod
from typing import Optional
from fastapi import Request


class AuthProvider(ABC):
    @abstractmethod
    def validate(self, request: Request) -> Optional[str]: ...


class NoAuthProvider(AuthProvider):
    def validate(self, request: Request) -> Optional[str]:
        return None
