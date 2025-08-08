# slack_adapter.py

from .base_adapter import ChannelAdapter
from slack_sdk import WebClient

class SlackAdapter(ChannelAdapter):
    """
    Slack 채널에서 메시지를 전송하는 어댑터입니다.
    """

    def __init__(self, client: WebClient, channel_id: str):
        self.client = client
        self.channel_id = channel_id

    def send_message(self, text: str):
        """
        Slack 채널에 메시지를 전송합니다.
        """
        self.client.chat_postMessage(channel=self.channel_id, text=text)
