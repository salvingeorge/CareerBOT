import os
import json
import sqlite3
from qibullet import SimulationManager
import pyttsx3
import threading
import time
from typing import Dict, List
from qibullet import PepperVirtual
import numpy as np
from speech_handler import SpeechSystem


class CareerCoachSystem:
    def __init__(self):
        # Initialize simulation
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)

        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.speech_system = SpeechSystem()
        self.speech_lock = threading.Lock()

        # Initialize database
        self.init_database()

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

    def speak(self, text: str, start_time: float = 0.0):
        """Make Pepper speak using both pyttsx3 and speech system"""
        time.sleep(start_time)
        print(f'Speaking careercoach method: {text}')

        with self.speech_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
                # Fallback to speech system if pyttsx3 fails
                # self.speech_system.speak(text, start_time)

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

        self.pepper.setAngles(["HeadYaw", "HeadPitch"], [0.0, 0.0], 1.0)

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

    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('career_coaching.db')
        self.cursor = self.conn.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_name TEXT,
                user_type TEXT,
                education TEXT,
                work_location TEXT,
                work_hours TEXT,
                stress_level TEXT,
                expected_salary TEXT,
                team_size TEXT,
                growth_speed TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                session_id INTEGER,
                career_title TEXT,
                match_percentage FLOAT,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            )
        ''')

        self.conn.commit()

    def conduct_career_assessment(self, user_info: Dict) -> Dict:
        """Conduct career assessment interview using speech recognition"""
        questions = [
            {
                "id": "education",
                "text": "What's your highest level of education?",
                "options": ["Bachelor's", "Master's", "PhD"]
            },
            {
                "id": "work_location",
                "text": "Do you prefer working remotely or in office?",
                "options": ["Remote", "In-Office", "Hybrid"]
            },
            {
                "id": "work_hours",
                "text": "What working hours do you prefer?",
                "options": ["Standard 9-5", "Flexible Hours", "Shift Work"]
            },
            {
                "id": "stress_level",
                "text": "What level of work stress can you handle?",
                "options": ["Low", "Moderate", "High"]
            },
            {
                "id": "expected_salary",
                "text": "What's your expected salary range?",
                "options": ["Entry Level", "Mid Range", "Senior Level"]
            },
            {
                "id": "team_size",
                "text": "What team size do you prefer working in?",
                "options": ["Small Team", "Medium Team", "Large Team"]
            },
            {
                "id": "growth_speed",
                "text": "How fast do you want your career to grow?",
                "options": ["Steady Pace", "Moderate Growth", "Fast Track"]
            }
        ]

        responses = {'user_name': user_info['name'], 'user_type': user_info['user_type']}

        for question in questions:
            # REMOVED: self.speech_system.speak(question['text'])
            response = self.speech_system.handle_question(question['text'], question['options'])
            responses[question['id']] = response

            # Visual feedback
            self.nod()
            time.sleep(0.5)

        return responses

    def generate_recommendations(self, responses: Dict) -> List[Dict]:
        """Generate career recommendations"""
        careers = [
            'Software Developer',
            'Data Scientist',
            'Product Manager',
            'UX Designer'
        ]

        recommendations = []
        for career in careers:
            match_score = np.random.randint(70, 100)  # Simulated matching
            recommendations.append({
                'title': career,
                'match': match_score
            })

        return sorted(recommendations, key=lambda x: x['match'], reverse=True)[:3]

    def present_recommendations(self, recommendations: List[Dict]):
        """Present career recommendations using speech"""
        self.speak("Based on your preferences, here are your top career matches:")

        for i, rec in enumerate(recommendations, 1):
            recommendation_text = f"{i}. {rec['title']} with {rec['match']}% match"
            self.speak(recommendation_text)
            time.sleep(0.5)

        # Add voice interaction for more details
        feedback_prompt = "Would you like to know more about any of these careers?"
        if self.speech_system.get_voice_input(feedback_prompt):
            self.speak("Please say the name of the career you'd like to know more about.")
            career_choice = self.speech_system.get_voice_input()
            if career_choice:
                for rec in recommendations:
                    if rec['title'].lower() in career_choice.lower():
                        self.provide_career_details(rec['title'])
                        break

    def provide_career_details(self, career_title: str):
        """Provide additional details about a specific career"""
        details = {
            'Software Developer': "Software developers create computer programs and applications. They need strong problem-solving skills and programming knowledge.",
            'Data Scientist': "Data scientists analyze complex data sets to help businesses make better decisions. Strong mathematics and statistics skills are essential.",
            'Product Manager': "Product managers oversee product development and strategy. They need excellent leadership and communication skills.",
            'UX Designer': "UX designers create user-friendly interfaces and experiences. They combine creativity with user research skills."
        }

        if career_title in details:
            self.speak(details[career_title])

    def save_session(self, responses: Dict, recommendations: List[Dict]):
        """Save session to database"""
        try:
            self.cursor.execute('''
                INSERT INTO sessions (
                    user_name, user_type, education, work_location
                ) VALUES (?, ?, ?, ?)
            ''', (
                responses['user_name'],
                responses['user_type'],
                responses.get('education', ''),
                responses.get('work_location', '')
            ))

            session_id = self.cursor.lastrowid

            for rec in recommendations:
                self.cursor.execute('''
                    INSERT INTO recommendations (session_id, career_title, match_percentage)
                    VALUES (?, ?, ?)
                ''', (session_id, rec['title'], rec['match']))

            self.conn.commit()
            print("Session saved successfully!")

        except Exception as e:
            print(f"Error saving session: {e}")
            self.conn.rollback()

    def start_interaction_with_user(self, user_info: Dict, greeting: str):
        """Start interaction with a detected user"""
        try:
            # Initial greeting
            self.execute_gesture('wave')
            self.speak(greeting)

            # Conduct assessment
            responses = self.conduct_career_assessment(user_info)

            # Generate and present recommendations
            recommendations = self.generate_recommendations(responses)
            self.present_recommendations(recommendations)

            # Save session
            self.save_session(responses, recommendations)

            # Farewell
            self.speak(f"Thank you for using our career coaching service, {user_info['name']}!")
            self.execute_gesture('wave')

        except Exception as e:
            print(f"Error in interaction: {e}")
            self.speak("I apologize, but there seems to be an error. Let's try again later.")

    def execute_gesture(self, gesture_name: str):
        """Execute a specific gesture"""
        if gesture_name == 'wave':
            self.wave()
        elif gesture_name == 'nod':
            self.nod()
        elif gesture_name == 'happy_swirl':
            self.happy_swirl()

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.conn.close()
        except:
            pass

        try:
            self.simulation_manager.stopSimulation(self.client)
        except:
            pass