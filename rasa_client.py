import requests
import json
from typing import Dict, List, Optional

class SyncRasaClient:
    def __init__(self, rasa_url: str = "http://localhost:5005"):
        self.rasa_url = rasa_url.rstrip('/')
        self.session = requests.Session()

    def send_message(self, message: str, sender_id: str) -> str:
        """Send a message to RASA and get response"""
        try:
            response = self.session.post(
                f"{self.rasa_url}/webhooks/rest/webhook",
                json={"sender": sender_id, "message": message}
            )
            if response.status_code == 200:
                responses = response.json()
                # Combine all text responses
                return " ".join([r.get('text', '') for r in responses if 'text' in r])
            else:
                print(f"Error from RASA server: {response.status_code}")
                return "I'm having trouble processing that right now."
        except Exception as e:
            print(f"Error communicating with RASA: {e}")
            return "Sorry, I'm unable to process your request at the moment."

    def is_server_running(self) -> bool:
        """Check if RASA server is running"""
        try:
            response = self.session.get(f"{self.rasa_url}/status")
            return response.status_code == 200
        except:
            return False