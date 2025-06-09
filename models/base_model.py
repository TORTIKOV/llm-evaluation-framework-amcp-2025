from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def generate(self, document: str, prompt: str):
        pass
