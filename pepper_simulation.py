import os
import json
import sqlite3
import threading
import time
import requests
from qibullet import SimulationManager
import pyttsx3
from typing import Dict, List, Optional
from qibullet import PepperVirtual
import numpy as np


class CareerCoachSystem:
    def __init__(self):
        # Initialize RASA connection
        self.base_url = "http://localhost:5005/webhooks/rest/webhook"
        self.sender_id = None

        # Initialize simulation
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)

        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.speech_lock = threading.Lock()

        # Default positions
        self.default_position = {
            "RShoulderPitch": 1.0,
            "RShoulderRoll": -0.1,
            "RElbowRoll": 0.3,
            "RElbowYaw": 0.0,
            "RWristYaw": 0.0,
            "RHand": 0.0,
            "HeadPitch": 0.0,
            "HeadYaw": 0.0
        }

        self.conversation_active = False

    def speak(self, text: str):
        """Make Pepper speak"""
        print(f'Pepper: {text}')
        with self.speech_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")

    def wave(self, duration: float = 3.0):
        """Wave gesture"""
        joints = ["RShoulderPitch", "RShoulderRoll", "RElbowRoll", "RElbowYaw"]
        positions = [
            [-0.2, -0.3, 1.2, 1.0],
            [-0.2, -0.3, 1.2, 1.5],
            [-0.2, -0.3, 1.2, 0.6]
        ]

        for position in positions:
            self.pepper.setAngles(joints, position, 1.0)
            time.sleep(duration / len(positions))

        self.reset_position()

    def nod(self, duration: float = 1.5):
        """Nodding gesture"""
        positions = [0.0, 0.3, 0.0, 0.3, 0.0]
        for position in positions:
            self.pepper.setAngles(["HeadPitch"], [position], 1.0)
            time.sleep(duration / len(positions))

    def happy_swirl(self, duration: float = 1.5):
        """Happy swirl gesture"""
        steps = 5
        radius = 0.3

        for i in range(steps):
            x = radius * np.cos(i * 2 * np.pi / steps)
            y = radius * np.sin(i * 2 * np.pi / steps)

            self.pepper.setAngles("RShoulderPitch", -1.0, 1.0)
            self.pepper.setAngles("LShoulderPitch", -1.0, 1.0)

            self.pepper.moveTo(x, y, 0, frame=PepperVirtual.FRAME_WORLD)
            time.sleep(duration / steps)

        self.reset_position()

    def reset_position(self):
        """Reset to default position"""
        joints = list(self.default_position.keys())
        angles = list(self.default_position.values())
        self.pepper.setAngles(joints, angles, 1.0)

    def _send_to_rasa(self, message: str) -> List[Dict]:
        """Send message to RASA and get response"""
        payload = {
            "sender": self.sender_id,
            "message": message
        }

        try:
            response = requests.post(self.base_url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error from RASA server: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with RASA: {e}")
            return []

    def _handle_rasa_response(self, responses: List[Dict]):
        """Handle RASA response with appropriate gestures"""
        for response in responses:
            if 'text' in response:
                text = response['text'].lower()

                # Perform gesture based on message content
                if any(word in text for word in ['hello', 'hi', 'greet']):
                    self.wave()
                elif any(word in text for word in ['goodbye', 'bye']):
                    self.wave()
                elif any(word in text for word in ['understand', 'ok', 'got it']):
                    self.nod()
                elif any(word in text for word in ['great', 'excellent', 'perfect']):
                    self.happy_swirl()

                # Speak the response
                self.speak(response['text'])

    def start_interaction(self, user_info: Dict):
        """Start interaction with a user"""
        self.sender_id = f"user_{user_info['name']}_{int(time.time())}"
        self.conversation_active = True

        try:
            # Initial greeting
            responses = self._send_to_rasa("hello")
            self._handle_rasa_response(responses)

            while self.conversation_active:
                # Get user input (for now, using text input)
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    self.conversation_active = False
                    break

                # Send to RASA and handle response
                responses = self._send_to_rasa(user_input)
                self._handle_rasa_response(responses)

        except KeyboardInterrupt:
            print("\nInteraction interrupted by user")
        except Exception as e:
            print(f"Error in interaction: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.conversation_active = False
        self.speak("Goodbye! Have a great day!")
        self.wave()
        try:
            self.simulation_manager.stopSimulation(self.client)
        except:
            pass


def main():
    career_coach = CareerCoachSystem()

    try:
        # For testing, we'll use a mock user
        user_info = {
            "name": "Test User",
            "user_type": "regular_user"
        }
        career_coach.start_interaction(user_info)
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        career_coach.cleanup()


if __name__ == "__main__":
    main()