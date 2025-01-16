import cv2
import numpy as np
import os
from datetime import datetime
import time


class FaceDetectionSystem:
    def __init__(self, confidence_threshold=0.6):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        except AttributeError:
            from cv2 import face
            self.face_recognizer = face.LBPHFaceRecognizer_create()

        self.confidence_threshold = confidence_threshold
        self.profiles_dir = "user_profiles"
        self.current_user_id = 0
        self.labels = {}  # Map of label to name
        self.load_profiles()
        print(f"Initialized with labels: {self.labels}")  # Debug print

    def load_profiles(self):
        """Load existing user profiles and train recognizer"""
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
            print("Created new profiles directory")  # Debug print
            return

        faces = []
        labels = []
        label_counter = 0

        print(f"Loading profiles from {self.profiles_dir}")  # Debug print
        for username in os.listdir(self.profiles_dir):
            user_dir = os.path.join(self.profiles_dir, username)
            if os.path.isdir(user_dir):
                self.labels[label_counter] = username
                print(f"Loading profile for {username}")  # Debug print
                for img_name in os.listdir(user_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(user_dir, img_name)
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if face_img is not None:  # Check if image was loaded successfully
                            faces.append(face_img)
                            labels.append(label_counter)
                            print(f"Loaded image: {img_path}")  # Debug print
                label_counter += 1

        if faces:  # Only train if we have existing profiles
            print(f"Training recognizer with {len(faces)} images")  # Debug print
            try:
                self.face_recognizer.train(faces, np.array(labels))
                print("Training completed successfully")  # Debug print
            except Exception as e:
                print(f"Error during training: {str(e)}")  # Debug print
            self.current_user_id = label_counter
        else:
            print("No faces found to train")  # Debug print

    def create_new_profile(self, frame, cap, name):
        """Create a new user profile with multiple captured photos"""
        user_dir = os.path.join(self.profiles_dir, name)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            print(f"Created directory for {name}")

        print("Taking multiple photos. Please move your face slightly between captures...")
        photos_taken = 0
        photos_needed = 10
        last_save_time = time.time()

        while photos_taken < photos_needed:
            # Capture new frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            # Detect face in new frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                cv2.putText(frame, "No face detected! Please adjust position",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Face Detection', frame)
                cv2.waitKey(1)
                continue

            # Get the largest face (closest to camera)
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            face_roi = frame[y:y + h, x:x + w]

            # Add a delay between captures
            current_time = time.time()
            if current_time - last_save_time >= 2.0:  # 2 second delay between captures
                # Save the current face image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{photos_taken}"
                img_path = os.path.join(user_dir, f"{timestamp}.jpg")

                # Prepare and save the image
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (200, 200))
                cv2.imwrite(img_path, gray_face)

                print(f"Saved photo {photos_taken + 1}/{photos_needed}")
                photos_taken += 1
                last_save_time = current_time

            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display feedback
            remaining_time = max(0, 2 - (current_time - last_save_time))
            cv2.putText(frame,
                        f"Photo {photos_taken}/{photos_needed} Next capture in: {remaining_time:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Face Detection', frame)

            # Check for 'q' key to cancel
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Profile creation cancelled")
                return False

        print("Profile creation complete!")
        # Update recognizer
        self.load_profiles()
        return True

    def detect_and_recognize(self, frame):
        """Detect and recognize faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_roi_gray = gray[y:y + h, x:x + w]

            # Ensure consistent size for recognition
            face_roi_gray = cv2.resize(face_roi_gray, (200, 200))

            if hasattr(self.face_recognizer, 'predict'):
                try:
                    label, confidence = self.face_recognizer.predict(face_roi_gray)
                    print(f"Recognition result - Label: {label}, Confidence: {confidence}")  # Debug print
                    if confidence < self.confidence_threshold * 100:
                        name = self.labels.get(label, "Unknown")
                        results.append({
                            'bbox': (x, y, w, h),
                            'name': name,
                            'confidence': confidence,
                            'recognized': True
                        })
                    else:
                        results.append({
                            'bbox': (x, y, w, h),
                            'name': "Unknown",
                            'confidence': confidence,
                            'recognized': False
                        })
                except Exception as e:
                    print(f"Recognition error: {str(e)}")  # Debug print
                    results.append({
                        'bbox': (x, y, w, h),
                        'name': "Unknown",
                        'confidence': 0,
                        'recognized': False
                    })

        return results


def main():
    print("Starting Face Detection System...")
    face_system = FaceDetectionSystem(confidence_threshold=0.8)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = face_system.detect_and_recognize(frame)

        for result in results:
            x, y, w, h = result['bbox']
            color = (0, 255, 0) if result['recognized'] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text = f"{result['name']} ({result['confidence']:.1f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if not result['recognized']:
                cv2.putText(frame, "Press 'N' to create new profile", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Face Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            name = input("Enter name for new profile: ")
            if name:
                if face_system.create_new_profile(frame, cap, name):
                    print(f"Profile created successfully for {name}")
                else:
                    print("Failed to create profile")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()