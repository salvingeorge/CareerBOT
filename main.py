import cv2
import threading
import time
from face_detection_system import FaceDetectionSystem
from face_expression_system import ExpressionSystem
from pepper_simulation import CareerCoachSystem
from typing import Dict, List, Any, Optional
from queue import Queue
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import requests


class IntegratedSystem:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize RASA connection
        self.rasa_url = "http://localhost:5005/webhooks/rest/webhook"
        self.verify_rasa_connection()

        self.logger.info("Initializing Face Detection System...")
        self.face_system = FaceDetectionSystem(confidence_threshold=0.8)

        self.logger.info("Initializing Expression System...")
        self.expression_system = ExpressionSystem()

        self.logger.info("Initializing Pepper System...")
        self.pepper_system = CareerCoachSystem()

        # Thread-safe containers
        self.interaction_queue = Queue()
        self.current_user: Optional[Dict] = None
        self._interaction_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()

        # State tracking
        self.interaction_active = False
        self.expression_history = []

        # Initialize event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.logger.info("All systems initialized successfully")

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def stop(self):
        self._running.clear()

    def process_face_detection(self, frame):
        """Process face detection results and trigger Pepper interactions"""
        try:
            results = self.face_system.detect_and_recognize(frame)
            self.logger.debug(f"Face detection results: {results}")

            with self._interaction_lock:
                if not self.interaction_active:
                    for result in results:
                        if result['recognized']:
                            self.logger.info(f"Recognized face: {result['name']}")
                            self.interaction_active = True
                            user_info = {
                                'name': result['name'],
                                'user_type': result['user_type']
                            }
                            self.interaction_queue.put(user_info)
                            return True
            return False

        except Exception as e:
            self.logger.error(f"Error in face detection: {str(e)}", exc_info=True)
            return False

    def verify_rasa_connection(self):
        """Verify RASA server is running"""
        try:
            test_message = {
                "sender": "test",
                "message": "hello"
            }
            response = requests.post(self.rasa_url, json=test_message)
            if response.status_code == 200:
                self.logger.info("RASA server connection verified")
            else:
                raise ConnectionError(f"RASA server returned status code {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.logger.error("Cannot connect to RASA server. Please ensure it's running with:")
            self.logger.error("rasa run -m models --enable-api --cors \"*\"")
            raise

    def send_to_rasa(self, user_id: str, message: str) -> list:
        """Send message to RASA and get response"""
        try:
            payload = {
                "sender": user_id,
                "message": message
            }
            response = requests.post(self.rasa_url, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"RASA error: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"RASA communication error: {e}")
            return []
    def handle_pepper_interaction(self):
        """Handle the Pepper robot interaction with detected users"""
        self.logger.info("Starting Pepper interaction handler thread")

        while self._running.is_set():
            try:
                if not self.interaction_queue.empty():
                    user_info = self.interaction_queue.get()
                    self.logger.info(f"\nStarting new interaction with {user_info['name']}")

                    with self._interaction_lock:
                        self.current_user = user_info
                        user_id = f"user_{user_info['name']}_{int(time.time())}"

                        # Initial greeting based on user type
                        greeting = ("Welcome back!" if user_info['user_type'] == "team_member"
                                    else "Hello! I'm your career coach!")

                        # Start RASA conversation
                        responses = self.send_to_rasa(user_id, "hello")
                        for response in responses:
                            if 'text' in response:
                                self.pepper_system.speak(response['text'])
                                time.sleep(1)  # Give time for speech

                        # Main interaction loop
                        while self.interaction_active and self._running.is_set():
                            # Get user input (replace with speech recognition later)
                            user_input = input("\nYou: ").strip()

                            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                                break

                            # Send to RASA and handle response
                            responses = self.send_to_rasa(user_id, user_input)
                            for response in responses:
                                if 'text' in response:
                                    self.pepper_system.speak(response['text'])
                                    # Add appropriate gesture based on response
                                    if any(word in response['text'].lower()
                                           for word in ['hello', 'hi', 'welcome']):
                                        self.pepper_system.wave()
                                    elif any(word in response['text'].lower()
                                             for word in ['goodbye', 'bye']):
                                        self.pepper_system.wave()
                                    elif any(word in response['text'].lower()
                                             for word in ['understand', 'ok', 'got it']):
                                        self.pepper_system.nod()

            except Exception as e:
                self.logger.error(f"Error in interaction handler: {e}", exc_info=True)
            finally:
                with self._interaction_lock:
                    self.interaction_active = False
                    self.current_user = None

                time.sleep(0.1)


    def analyze_user_engagement(self, expression_results):
        """Analyze user engagement based on expressions with improved thresholds"""
        if not expression_results:
            return "unknown"

        try:
            latest_expression = expression_results[0]['expression']
            confidence = expression_results[0]['confidence']

            # More nuanced engagement analysis
            if latest_expression == 'happy':
                if confidence > 0.8:
                    return "highly_engaged"
                elif confidence > 0.6:
                    return "engaged"
            elif latest_expression == 'sad':
                if confidence > 0.7:
                    return "disengaged"
                elif confidence > 0.5:
                    return "slightly_disengaged"

            return "neutral"

        except Exception as e:
            self.logger.error(f"Error in engagement analysis: {str(e)}")
            return "unknown"

    def update_display(self, frame, face_results, expression_results):
        """Update the display with current results and improved visualization"""
        try:
            # Track expressions during interaction
            if self.interaction_active and expression_results:
                self.expression_history.append(expression_results[0])
                if len(self.expression_history) > 50:  # Limit history size
                    self.expression_history.pop(0)

                engagement = self.analyze_user_engagement(expression_results)
                engagement_color = {
                    "highly_engaged": (0, 255, 0),
                    "engaged": (0, 200, 0),
                    "neutral": (255, 255, 0),
                    "slightly_disengaged": (200, 0, 0),
                    "disengaged": (0, 0, 255),
                    "unknown": (128, 128, 128)
                }[engagement]

                cv2.putText(frame, f"Engagement: {engagement}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, engagement_color, 2)

            # Display face detection results with improved visibility
            for result in face_results:
                x, y, w, h = result['bbox']
                # Different colors for different user types
                if result.get('recognized'):
                    color = ((0, 255, 0) if result['user_type'] == "team_member"
                             else (255, 165, 0))
                    confidence_text = f" ({result['confidence']:.1f}%)"
                else:
                    color = (0, 0, 255)
                    confidence_text = ""

                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Add text with better positioning and background
                text = f"{result['name']}{confidence_text}"
                if result.get('recognized'):
                    text += f" - {result['user_type'].replace('_', ' ').title()}"

                # Add background rectangle for text
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display system status
            status = "Active Interaction" if self.interaction_active else "Waiting for User"
            cv2.putText(frame, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Career Coach System', frame)
            return cv2.waitKey(1)

        except Exception as e:
            self.logger.error(f"Error in display update: {str(e)}", exc_info=True)
            return -1

    def run(self):
        """Main method to run the system with improved error handling"""
        self.logger.info("\nStarting Integrated Career Coach System...")

        try:
            # Start Pepper interaction handler in a separate thread
            self.logger.info("Starting interaction handler thread...")
            interaction_thread = threading.Thread(target=self.handle_pepper_interaction)
            interaction_thread.daemon = True
            interaction_thread.start()
            self.logger.info("Interaction handler thread started")

            # Initialize video capture with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    self.logger.info("Camera initialized successfully")
                    break
                self.logger.warning(f"Camera initialization attempt {attempt + 1} failed")
                time.sleep(1)
            else:
                self.logger.error("Could not initialize camera after multiple attempts")
                return

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to grab frame")
                    time.sleep(0.1)
                    continue

                # Process detection and update display
                face_results = self.face_system.detect_and_recognize(frame)
                expression_results = self.expression_system.process_frame(frame)

                key = self.update_display(frame, face_results, expression_results)

                # Process detection for Pepper interaction
                if not self.interaction_active:
                    if self.process_face_detection(frame):
                        self.logger.info("New face detected and queued for interaction")

                # Handle user input
                if key == ord('q'):
                    self.logger.info("Quit command received")
                    break
                elif key == ord('n') and not self.interaction_active:
                    name = input("Enter name for new profile: ")
                    if name:
                        if self.face_system.create_new_profile(frame, cap, name):
                            self.logger.info(f"Profile created successfully for {name}")
                        else:
                            self.logger.error("Failed to create profile")

        except Exception as e:
            self.logger.error(f"Error in main system loop: {str(e)}", exc_info=True)

        finally:
            self.stop()
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            self.pepper_system.cleanup()
            self.logger.info("System shutdown completed")

    def cleanup(self):
        """Cleanup resources and stop event loop"""
        try:
            self._running.clear()
            if hasattr(self, 'pepper_system'):
                self.pepper_system.cleanup()
            if hasattr(self, 'loop'):
                self.loop.stop()
                self.loop.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
        finally:
            cv2.destroyAllWindows()


def main():
    logging.info("\n=== Starting Career Coach System ===")
    system = None
    try:
        system = IntegratedSystem()
        system.run()
    except Exception as e:
        logging.error(f"Critical error in main: {str(e)}", exc_info=True)
    finally:
        if system:
            system.stop()
        cv2.destroyAllWindows()
        logging.info("System shutdown completed")


if __name__ == "__main__":
    main()
