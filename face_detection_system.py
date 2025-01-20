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

        # Define team members
        self.team_members = {
            "Pallavi Aithal Narayan",
            "Salvin George",
            "Darnaux Maxime",
            "Othmane Elmekaoui"
        }
        self.user_types = {}  # Map of label to user type (team_member or regular_user)
        self.load_profiles()
        print(f"Initialized with labels: {self.labels}")  # Debug print

    def is_team_member(self, input_name):
        """Check if a name belongs to a team member using simple string operations"""
        input_parts = set(input_name.lower().split())

        for full_name in self.team_members:
            member_parts = set(full_name.lower().split())
            # Check if any part of the input name matches any part of the team member's name
            if input_parts & member_parts:  # intersection of sets
                return True
        return False

    def get_full_name(self, input_name):
        """Get the full name if it's a team member, otherwise return the original name"""
        input_parts = set(input_name.lower().split())

        for full_name in self.team_members:
            member_parts = set(full_name.lower().split())
            # If there's any overlap in the name parts, return the full name
            if input_parts & member_parts:
                return full_name
        return input_name

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
                full_name = self.get_full_name(username)
                self.labels[label_counter] = full_name
                # Determine if the user is a team member
                self.user_types[label_counter] = "team_member" if self.is_team_member(username) else "regular_user"
                print(f"Loading profile for {full_name} ({self.user_types[label_counter]})")  # Debug print

                for img_name in os.listdir(user_dir):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(user_dir, img_name)
                        face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if face_img is not None:
                            faces.append(face_img)
                            labels.append(label_counter)
                            print(f"Loaded image: {img_path}")  # Debug print
                label_counter += 1

        if faces:
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
        # Get the full name if it's a team member, otherwise use the provided name
        full_name = self.get_full_name(name)
        user_dir = os.path.join(self.profiles_dir, full_name)

        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
            print(f"Created directory for {full_name}")

        # Determine if the new user is a team member
        is_team_member = self.is_team_member(name)
        user_type = "team_member" if is_team_member else "regular_user"
        print(f"Creating profile for {full_name} as {user_type}")

        print("Taking multiple photos. Please move your face slightly between captures...")
        photos_taken = 0
        photos_needed = 10
        last_save_time = time.time()

        while photos_taken < photos_needed:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                cv2.putText(frame, "No face detected! Please adjust position",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Face Detection', frame)
                cv2.waitKey(1)
                continue

            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            face_roi = frame[y:y + h, x:x + w]

            current_time = time.time()
            if current_time - last_save_time >= 2.0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{photos_taken}"
                img_path = os.path.join(user_dir, f"{timestamp}.jpg")

                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (200, 200))
                cv2.imwrite(img_path, gray_face)

                print(f"Saved photo {photos_taken + 1}/{photos_needed}")
                photos_taken += 1
                last_save_time = current_time

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            remaining_time = max(0, 2 - (current_time - last_save_time))
            status_text = f"Photo {photos_taken}/{photos_needed} Next capture in: {remaining_time:.1f}s"
            if is_team_member:
                status_text += f" (Team Member: {full_name})"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Profile creation cancelled")
                return False

        print(f"Profile creation complete for {full_name} ({user_type})!")
        self.load_profiles()
        return True

    def detect_and_recognize(self, frame):
        """Detect and recognize faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        results = []
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            if hasattr(self.face_recognizer, 'predict'):
                try:
                    label, confidence = self.face_recognizer.predict(face_roi)
                    # print(f"Recognition result - Label: {label}, Confidence: {confidence}")  # Debug print

                    if confidence < self.confidence_threshold * 100:
                        name = self.labels.get(label, "Unknown")
                        user_type = self.user_types.get(label, "unknown")
                        results.append({
                            'bbox': (x, y, w, h),
                            'name': name,
                            'confidence': confidence,
                            'recognized': True,
                            'user_type': user_type
                        })
                    else:
                        results.append({
                            'bbox': (x, y, w, h),
                            'name': "Unknown",
                            'confidence': confidence,
                            'recognized': False,
                            'user_type': "unknown"
                        })
                except Exception as e:
                    print(f"Recognition error: {str(e)}")  # Debug print
                    results.append({
                        'bbox': (x, y, w, h),
                        'name': "Unknown",
                        'confidence': 0,
                        'recognized': False,
                        'user_type': "unknown"
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

            # Different colors for team members vs regular users
            if result['recognized']:
                if result['user_type'] == "team_member":
                    color = (0, 255, 0)  # Green for team members
                else:
                    color = (255, 165, 0)  # Orange for regular users
            else:
                color = (0, 0, 255)  # Red for unknown faces

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display name, confidence, and user type
            text = f"{result['name']} ({result['confidence']:.1f}%)"
            if result['recognized']:
                text += f" - {result['user_type'].replace('_', ' ').title()}"
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