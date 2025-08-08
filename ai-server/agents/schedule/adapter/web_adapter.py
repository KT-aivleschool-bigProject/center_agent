# web_adapter.py

from .base_adapter import ChannelAdapter

class WebAdapter(ChannelAdapter):
    """
    Web 채팅 인터페이스에서 메시지를 저장하는 어댑터입니다. (Web 챗봇용)
    """

    def __init__(self):
        self.messages = []

    def send_message(self, text: str):
        """
        메시지를 내부 리스트에 저장합니다.
        """
        self.messages.append(text)

    def get_response(self):
        """
        마지막 메시지를 반환합니다.
        """
        return self.messages[-1] if self.messages else "응답 없음"
