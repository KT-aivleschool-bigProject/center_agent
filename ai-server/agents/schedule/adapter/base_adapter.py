# base_adapter.py

from abc import ABC, abstractmethod

class ChannelAdapter(ABC):
    """
    모든 채널(Slack, Web 등)을 위한 공통 메시지 전송 인터페이스입니다.
    """

    @abstractmethod
    def send_message(self, text: str):
        """
        사용자에게 메시지를 전송합니다.
        """
        pass
