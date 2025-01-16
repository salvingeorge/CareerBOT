import cv2
import threading
import time
from face_detection_system import FaceDetectionSystem
from pepper_simulation import CareerCoachSystem
from queue import Queue
import json


class IntegratedSystem:
    def __init__(self):
        self.face_system = FaceDetectionSystem(confidence_threshold=0.8)
        self.pepper_system = CareerCoachSystem()
        self.interaction_queue = Queue()
        self.current_user = None
        self.interaction_active = False

    def process_face_detection(self, frame):
        """Process face detection results and trigger Pepper interactions"""
        results = self.face_system.detect_and_recognize(frame)

        for result in results:
            if result['recognized']:
                if not self.interaction_active:
                    self.interaction_active = True
                    user_info = {
                        'name': result['name'],
                        'user_type': result['user_type']
                    }
                    self.interaction_queue.put(user_info)
                    return True
        return False

    def handle_pepper_interaction(self):
        """Handle the Pepper robot interaction with detected users"""
        while True:
            if not self.interaction_queue.empty():
                user_info = self.interaction_queue.get()
                print(f"Starting interaction with {user_info['name']}")

                # Initialize greeting based on user type
                if user_info['user_type'] == "team_member":
                    greeting = f"Welcome back, {user_info['name']}! Nice to see you again!"
                else:
                    greeting = f"Hello {user_info['name']}! I'm your career coach!"

                # Start the Pepper interaction
                self.pepper_system.start_interaction_with_user(user_info, greeting)

                # Reset interaction flag after completion
                self.interaction_active = False

            time.sleep(0.1)

    def run(self):
        """Main loop to run the integrated system"""
        print("Starting Integrated Career Coach System...")
        cap = cv2.VideoCapture(0)

        # Start Pepper interaction handler in a separate thread
        pepper_thread = threading.Thread(target=self.handle_pepper_interaction)
        pepper_thread.daemon = True
        pepper_thread.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process face detection
            results = self.face_system.detect_and_recognize(frame)

            # Display face detection results
            for result in results:
                x, y, w, h = result['bbox']

                # Color coding based on user type
                if result['recognized']:
                    if result['user_type'] == "team_member":
                        color = (0, 255, 0)  # Green for team members
                    else:
                        color = (255, 165, 0)  # Orange for regular users
                else:
                    color = (0, 0, 255)  # Red for unknown faces

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Display recognition info
                text = f"{result['name']} ({result['confidence']:.1f}%)"
                if result['recognized']:
                    text += f" - {result['user_type'].replace('_', ' ').title()}"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Process detection results for Pepper interaction
            if not self.interaction_active:
                self.process_face_detection(frame)

            # Display system status
            status = "Active Interaction" if self.interaction_active else "Waiting for User"
            cv2.putText(frame, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Career Coach System', frame)

            # Handle user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and not self.interaction_active:
                name = input("Enter name for new profile: ")
                if name:
                    if self.face_system.create_new_profile(frame, cap, name):
                        print(f"Profile created successfully for {name}")
                    else:
                        print("Failed to create profile")

        cap.release()
        cv2.destroyAllWindows()
        self.pepper_system.cleanup()


def main():
    try:
        integrated_system = IntegratedSystem()
        integrated_system.run()
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()