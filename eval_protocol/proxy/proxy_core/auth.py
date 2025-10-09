from abc import ABC, abstractmethod
from typing import Optional


class AuthProvider(ABC):
    @abstractmethod
    def validate(self, api_key: Optional[str]) -> Optional[str]: ...


class NoAuthProvider(AuthProvider):
    def validate(self, api_key: Optional[str]) -> Optional[str]:
        return None
