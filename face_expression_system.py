import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import deque


class ExpressionSystem:
    def __init__(self, confidence_threshold=0.6):
        # Load only necessary cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        self.confidence_threshold = confidence_threshold
        # Using deque for more efficient history tracking
        self.expression_history = deque(maxlen=10)  # Increased history size for smoother detection

    def detect_smile(self, face_roi: np.ndarray) -> Tuple[bool, float]:
        """Detect smile and calculate intensity"""
        smiles = self.smile_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.3,  # Made more sensitive
            minNeighbors=15,  # Reduced to detect more smiles
            minSize=(20, 20)  # Smaller minimum size for detection
        )

        if len(smiles) > 0:
            # Get the largest smile detection
            largest_smile = max(smiles, key=lambda rect: rect[2] * rect[3])
            _, _, w, h = largest_smile

            # Calculate smile intensity relative to face size
            smile_intensity = (w * h) / (face_roi.shape[0] * face_roi.shape[1])
            return True, smile_intensity
        return False, 0.0

    def determine_expression(self, smile_detected: bool, smile_intensity: float) -> Tuple[str, float]:
        """Determine if expression is happy or sad based on smile detection"""
        if smile_detected:
            if smile_intensity > 0.1:  # Strong smile
                return 'happy', min(smile_intensity * 2, 1.0)
            elif smile_intensity > 0.05:  # Slight smile
                return 'slight_happy', smile_intensity

        # No smile detected implies neutral/sad
        # The longer we don't detect a smile, the more confident we are about sadness
        sad_confidence = min(len([e for e in self.expression_history if e == 'sad']) * 0.1, 0.8)
        return 'sad', sad_confidence

    def smooth_expression(self, current_expression: str) -> str:
        """Smooth expression detection using historical data"""
        self.expression_history.append(current_expression)

        # Count expressions in recent history
        hist_count = dict()
        for expr in self.expression_history:
            hist_count[expr] = hist_count.get(expr, 0) + 1

        # Return most common expression
        return max(hist_count.items(), key=lambda x: x[1])[0]

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Process a frame and return face detection results with expressions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        results = []
        for (x, y, w, h) in faces:
            try:
                face_roi = gray[y:y + h, x:x + w]

                # Detect smile
                has_smile, smile_intensity = self.detect_smile(face_roi)

                # Determine expression and confidence
                expression, confidence = self.determine_expression(has_smile, smile_intensity)

                # Smooth expression
                smoothed_expression = self.smooth_expression(expression)

                results.append({
                    'bbox': (x, y, w, h),
                    'expression': smoothed_expression,
                    'confidence': confidence,
                    'smile_intensity': smile_intensity
                })

            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        return results


def main():
    face_system = ExpressionSystem()
    cap = cv2.VideoCapture(0)

    print("Starting expression detection (Happy/Sad)...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = face_system.process_frame(frame)

        for result in results:
            x, y, w, h = result['bbox']

            # Color mapping
            color = {
                'happy': (0, 255, 0),  # Green
                'slight_happy': (0, 255, 128),  # Light green
                'sad': (0, 0, 255)  # Red
            }.get(result['expression'], (255, 255, 255))

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw expression and confidence
            label = f"{result['expression']} ({result['confidence']:.2f})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw smile intensity
            intensity_label = f"Smile intensity: {result['smile_intensity']:.2f}"
            cv2.putText(frame, intensity_label, (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Expression Detection (Happy/Sad)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()