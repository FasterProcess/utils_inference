from abc import ABC, abstractmethod
from typing import Any


class VideoStreamWriter(ABC):
    def __init__(
        self, save_path, writer=None, refer_file=None, loglevel="warning", **kwargs
    ):
        pass

    @abstractmethod
    def Write(self, data: bytes) -> bool:
        pass

    @abstractmethod
    def Close(self):
        pass

    def set_process(self, writer: Any):
        pass
