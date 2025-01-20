import os
import json
import sqlite3
from qibullet import SimulationManager
import pyttsx3
import threading
import time
from typing import Dict, List, Any, Optional
from qibullet import PepperVirtual
import numpy as np
from speech_handler import SpeechSystem
from bayesian_network import CareerBayesianNetwork
import asyncio
from rasa.core.agent import Agent
from rasa.shared.utils.io import json_to_string
from rasa.utils.endpoints import EndpointConfig
import requests
from rasa_client import SyncRasaClient
from dataclasses import dataclass
from enum import Enum, auto
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

class InteractionMode:
    VOICE = 'voice'
    TEXT = 'text'

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

        self.db_lock = threading.Lock()
        self._local = threading.local()
        # Initialize database
        self.init_database()
        self.conn = sqlite3.connect('career_coaching.db')
        self.cursor = self.conn.cursor()

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

        self.interaction_mode = InteractionMode.TEXT
        self.current_responses = {}

    def select_interaction_mode(self) -> str:
        self.speak("Would you like to proceed with voice interaction?")
        print("\nWould you like to proceed with voice interaction? (yes/no)")

        while True:
            choice = input("Enter 'yes' or 'no': ").strip().lower()

            if choice in ['yes', 'y']:
                self.speak("Voice interaction mode selected.")
                self.interaction_mode = InteractionMode.VOICE
                return self.interaction_mode
            elif choice in ['no', 'n']:
                self.speak("Text interaction mode selected.")
                self.interaction_mode = InteractionMode.TEXT
                return self.interaction_mode
            else:
                print("Invalid choice. Please enter 'yes' or 'no'.")

    def get_user_input(self, prompt: str, options: List[str] = None, validation: List[str] = None) -> str:
        retry_count = 0
        max_retries = 3

        def normalize_text(text):
            """Normalize text by converting to lowercase and removing special characters."""
            return text.lower().strip().replace("'", "")

        while retry_count < max_retries:
            print(f"\n{prompt}")
            if options:
                for i, option in enumerate(options, 1):
                    print(f"{i}. {option}")

            user_input = input("Your response: ").strip()
            response_normalized = normalize_text(user_input)

            # Normalize options and validation lists
            valid_options_normalized = [normalize_text(opt) for opt in options] if options else []
            validation_normalized = [normalize_text(val) for val in validation] if validation else []

            # Check against both options and validation
            if response_normalized in valid_options_normalized:
                return options[valid_options_normalized.index(response_normalized)]
            elif response_normalized in validation_normalized:
                # Return the first matching validation option if validation is provided
                return validation[validation_normalized.index(response_normalized)]

            print("Invalid response. Please choose from the available options.")
            retry_count += 1

        raise ValueError("Maximum retries exceeded for user input.")


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
        """Conduct career assessment interview using selected interaction mode"""
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
            self.speak(question['text'])
            response = self.get_user_input(
                question['text'],
                options=question['options'],
                validation=question['validation']
            )

            responses[question['id']] = response

            # Visual feedback
            self.nod()
            time.sleep(0.5)

        self.current_responses = responses

        return responses

    def generate_recommendations(self, responses: Dict) -> List[Dict]:
        """Generate career recommendations using Bayesian network"""
        try:
            # Get the Bayesian network instance
            network = CareerBayesianNetwork.get_instance()

            # Filter the responses to only include the fields needed for the Bayesian network
            preferences = {
                'education': responses.get('education'),
                'work_location': responses.get('work_location'),
                'work_hours': responses.get('work_hours'),
                'stress_level': responses.get('stress_level'),
                'expected_salary': responses.get('expected_salary'),
                'team_size': responses.get('team_size'),
                'growth_speed': responses.get('growth_speed')
            }

            # Get recommendations from the network
            recommendations = network.get_recommendations(preferences, top_n=3)

            if not recommendations:
                # Fallback to default recommendations if the network fails
                print("Warning: Bayesian network failed to provide recommendations, using fallback options")
                return [
                    {'title': 'Software Developer', 'match': 85},
                    {'title': 'Data Scientist', 'match': 80},
                    {'title': 'Product Manager', 'match': 75}
                ]

            return recommendations

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            # Return default recommendations in case of error
            return [
                {'title': 'Software Developer', 'match': 85},
                {'title': 'Data Scientist', 'match': 80},
                {'title': 'Product Manager', 'match': 75}
            ]

    def present_recommendations(self, recommendations: List[Dict]):
        """Present career recommendations using selected interaction mode"""
        try:
            # First present the recommendations
            self.speak("Based on your preferences, here are your top career matches:")

            for i, rec in enumerate(recommendations, 1):
                recommendation_text = f"{i}. {rec['title']} with {rec['match']}% match"
                self.speak(recommendation_text)
                print(recommendation_text)
                time.sleep(0.5)

            # First save the session before asking for more details
            try:
                self.save_session(self.current_responses, recommendations)
            except Exception as e:
                print(f"Warning: Could not save session: {e}")

            # Then handle the career details interaction
            more_details = self.get_user_input("Would you like to know more about any of these careers? (yes/no)")

            if more_details.lower() in ['yes', 'y']:
                career_prompt = "Please specify which career you'd like to know more about:"
                if self.interaction_mode == InteractionMode.TEXT:
                    print("\nAvailable careers:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"{i}. {rec['title']}")

                career_choice = self.get_user_input(career_prompt)

                # Match either by number or name
                selected_career = None
                try:
                    # Try to match by number first
                    choice_num = int(career_choice)
                    if 1 <= choice_num <= len(recommendations):
                        selected_career = recommendations[choice_num - 1]['title']
                except ValueError:
                    # If not a number, try to match by name
                    for rec in recommendations:
                        if rec['title'].lower() in career_choice.lower():
                            selected_career = rec['title']
                            break

                if selected_career:
                    self.provide_career_details(selected_career)
                else:
                    self.speak("I'm sorry, I couldn't find that career in the recommendations.")

        except Exception as e:
            print(f"Error in presenting recommendations: {e}")
            self.speak("I'm sorry, there was an error presenting the recommendations.")

    def provide_career_details(self, career_title: str):
        """Provide additional details about a specific career"""
        details = {
            'Software Developer': "Software developers create computer programs and applications. They need strong problem-solving skills and programming knowledge.",
            'Data Scientist': "Data scientists analyze complex data sets to help businesses make better decisions. Strong mathematics and statistics skills are essential.",
            'Project Manager': "Project managers oversee project planning, execution, and delivery. They need excellent organizational and leadership skills.",
            'UX Designer': "UX designers create user-friendly interfaces and experiences. They combine creativity with user research skills.",
            'Business Analyst': "Business analysts bridge the gap between business and IT, analyzing processes and recommending improvements.",
            'DevOps Engineer': "DevOps engineers manage the infrastructure and deployment pipelines. They need strong automation and systems knowledge."
        }

        if career_title in details:
            self.speak(details[career_title])
        else:
            self.speak(f"I'm sorry, I don't have detailed information about {career_title} at the moment.")

    def get_db_connection(self):
        """Get thread-local database connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect('career_coaching.db')
        return self._local.conn

    def get_db_cursor(self):
        """Get thread-local database cursor"""
        if not hasattr(self._local, 'cursor') or self._local.cursor is None:
            self._local.cursor = self.get_db_connection().cursor()
        return self._local.cursor

    def save_response(self, question: str, response: str):
        with self.db_lock:
            try:
                self.cursor.execute(
                    "INSERT INTO user_responses (question, response) VALUES (?, ?)", (question, response))
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Error saving response: {e}")

    def save_session(self, responses: Dict, recommendations: List[Dict]):
        """Save session to database with thread safety"""
        with self.db_lock:  # Use the lock when saving data
            try:
                conn = self.get_db_connection()
                cursor = self.get_db_cursor()

                cursor.execute('''
                    INSERT INTO sessions (
                        user_name, user_type, education, work_location, work_hours, 
                        stress_level, expected_salary, team_size, growth_speed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    responses['user_name'],
                    responses['user_type'],
                    responses.get('education', ''),
                    responses.get('work_location', ''),
                    responses.get('work_hours', ''),
                    responses.get('stress_level', ''),
                    responses.get('expected_salary', ''),
                    responses.get('team_size', ''),
                    responses.get('growth_speed', '')
                ))

                session_id = cursor.lastrowid

                for rec in recommendations:
                    cursor.execute('''
                        INSERT INTO recommendations (session_id, career_title, match_percentage)
                        VALUES (?, ?, ?)
                    ''', (session_id, rec['title'], rec['match']))

                conn.commit()
                print("Session saved successfully!")

            except Exception as e:
                print(f"Error saving session: {e}")
                if hasattr(self._local, 'conn') and self._local.conn:
                    self._local.conn.rollback()
                raise

    def reset_session(self):
        """Reset session data after interaction"""
        self.current_user = None
        self.current_responses = {}

    def start_interaction_with_user(self, user_info: Dict, greeting: str):
        """Start interaction with a detected user"""
        try:
            # Select interaction mode
            self.interaction_mode = self.select_interaction_mode()

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
            self.reset_session()

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

class MessageType(Enum):
    SYSTEM = auto()
    PEPPER = auto()
    USER = auto()
    DEBUG = auto()


class InteractionLogger:
    def __init__(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='career_coach.log'
        )
        self.logger = logging.getLogger(__name__)

        # Console handler for user interaction
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.console_handler)

    def format_message(self, message: str, msg_type: MessageType) -> str:
        """Format message based on type"""
        if msg_type == MessageType.PEPPER:
            return f"\nPepper: {message}\n"
        elif msg_type == MessageType.USER:
            return f"User: {message}"
        elif msg_type == MessageType.SYSTEM:
            return f"\n{'-' * 40}\n{message}\n{'-' * 40}\n"
        else:  # DEBUG
            return f"Debug: {message}"

    def log(self, message: str, msg_type: MessageType):
        """Log a message with appropriate formatting"""
        formatted_msg = self.format_message(message, msg_type)
        if msg_type == MessageType.DEBUG:
            self.logger.debug(formatted_msg)
        else:
            print(formatted_msg)
            self.logger.info(formatted_msg)

class RASAInterface:
    def __init__(self, base_url: str = "http://localhost:5005"):
        self.base_url = base_url
        self.webhook_url = f"{base_url}/webhooks/rest/webhook"

    def send_message(self, message: str, sender_id: str) -> List[Dict]:
        """Send a message to RASA and get response"""
        payload = {
            "sender": sender_id,
            "message": message
        }

        try:
            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with RASA server: {e}")
            return []


class RasaIntegration:
    def __init__(self, model_path: str = "./models"):
        """Initialize RASA integration"""
        self.model_path = model_path
        self.agent = None
        self.action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")

    async def initialize(self) -> bool:
        """Initialize the RASA agent asynchronously"""
        try:
            # Check if model path exists
            if not os.path.exists(self.model_path):
                print(f"Model path {self.model_path} does not exist")
                return False

            # Load the agent
            self.agent = Agent.load(
                self.model_path,
                action_endpoint=self.action_endpoint
            )

            if not self.agent:
                print("Failed to load RASA agent")
                return False

            return True
        except Exception as e:
            print(f"Error initializing RASA agent: {e}")
            return False

    async def get_response(self, message: str, sender_id: str) -> List[Dict[str, Any]]:
        """Get response from RASA for a message"""
        try:
            if not self.agent:
                initialized = await self.initialize()
                if not initialized:
                    return [{"text": "I'm having trouble initializing. Please try again later."}]

            responses = await self.agent.handle_text(
                text_message=message,
                sender_id=sender_id,
                output_channel=None
            )
            return responses if responses else [{"text": "I didn't get a response."}]

        except Exception as e:
            print(f"Error getting RASA response: {e}")
            return [{"text": "I'm having trouble processing that right now."}]


@dataclass
class CareerDetails:
    """Career details container"""
    title: str
    description: str
    skills: List[str]
    education: List[str]
    salary_range: str


class UserInteractionHandler:
    def __init__(self, speech_system, speech_lock, engine):
        self.speech_system = speech_system
        self.speech_lock = speech_lock
        self.engine = engine
        self.interaction_mode = InteractionMode.TEXT
        self.last_prompt = None
        self.response_timeout = 30

        # Common variations of responses
        self._common_variations = {
            "bachelor": ["bachelors", "bachelor", "bachelor's", "bs", "b.s.", "undergraduate"],
            "master": ["masters", "master", "master's", "ms", "m.s.", "postgraduate"],
            "phd": ["phd", "ph.d.", "doctorate", "doctoral"],
            "remote": ["remote", "remotely", "work from home", "wfh", "remote work"],
            "in-office": ["in office", "in-office", "office", "onsite", "on-site"],
            "hybrid": ["hybrid", "mixed", "both", "flexible"],
            "standard": ["standard", "9-5", "nine to five", "regular", "normal"],
            "flexible": ["flexible", "flex", "variable"],
            "shift": ["shift", "shifts", "shift work"],
            "low": ["low", "minimal", "light"],
            "moderate": ["moderate", "medium", "mid"],
            "high": ["high", "heavy", "intense"],
            "small": ["small", "tiny", "compact"],
            "medium": ["medium", "mid-size", "moderate size"],
            "large": ["large", "big", "huge"]
        }

    def normalize_input(self, user_input: str) -> str:
        """Normalize user input by removing special characters and converting to lowercase"""
        normalized = user_input.lower().strip()
        normalized = normalized.replace("'", "").replace(".", "")
        return normalized

    def normalize_text(text: str) -> str:
        """Normalize text by converting to lowercase and removing special characters."""
        return text.lower().strip().replace("'", "")

    def _find_best_match(self, user_input: str, valid_options: List[str]) -> Optional[str]:
        """Find the best matching option for user input"""
        normalized_input = self._normalize_input(user_input)

        # Try number matching first
        try:
            choice_num = int(user_input)
            if 1 <= choice_num <= len(valid_options):
                return valid_options[choice_num - 1]
        except ValueError:
            pass

        # Direct match with options
        for option in valid_options:
            if self._normalize_input(option) == normalized_input:
                return option

        # Try matching with common variations
        input_words = normalized_input.split()
        for word in input_words:
            for base_word, variations in self._common_variations.items():
                if word in variations:
                    # Find matching option
                    for option in valid_options:
                        if base_word in self._normalize_input(option):
                            return option

        return None

    def match_input_variation(self, user_input: str, valid_options: List[str]) -> Optional[str]:
        """Match user input against known variations"""
        normalized_input = self.normalize_input(user_input)

        # First check direct matches
        for option in valid_options:
            if normalized_input == self.normalize_input(option):
                return option

        # Then check variations
        for standard, variations in self.input_variations.items():
            if normalized_input in variations:
                # Find the matching option
                for option in valid_options:
                    if self.normalize_input(option) in [standard] + variations:
                        return option

        return None

    def speak_and_wait(self, text: str, start_time: float = 0.0) -> None:
        """Speak text and ensure it completes before continuing"""
        time.sleep(start_time)
        print(f'\nPepper: {text}')

        with self.speech_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")
                time.sleep(len(text) * 0.1)

    def _normalize_input(self, text: str) -> str:
        """Normalize input text for comparison"""
        return text.lower().strip().replace("'", "").replace("-", " ")

    def get_validated_input(self, prompt: str, options: List[str] = None, validation: List[str] = None) -> str:
        """Get and validate user input with normalization"""
        max_retries = 3
        retries = 0

        # Normalize options and validation for case-insensitive matching
        normalized_options = {opt.lower(): opt for opt in options} if options else {}
        normalized_validation = {val.lower(): val for val in validation} if validation else {}

        while retries < max_retries:
            # Display prompt and options
            print("\nUser Input Required:")
            print("-" * 40)
            print(f"{prompt}")

            if options:
                print("\nAvailable options:")
                for idx, option in enumerate(options, 1):
                    print(f"{idx}. {option}")

            try:
                # Get input
                response = input("\nYour response: ").strip()
                if not response:
                    print("\nPlease provide a response.")
                    retries += 1
                    continue

                # Normalize input
                response_normalized = response.lower()

                # Match with normalized options or validation
                if response_normalized in normalized_options:
                    return normalized_options[response_normalized]
                elif response_normalized in normalized_validation:
                    return normalized_validation[response_normalized]

                print("\nInvalid response. Please choose from the available options:")
                retries += 1

            except Exception as e:
                print(f"Error getting input: {e}")
                retries += 1

        # Fallback after retries
        if options:
            print("\nMaximum retries reached. Using the first option as default.")
            return options[0]
        raise ValueError("Failed to get valid input after multiple attempts.")

    def _debug_response(self, user_input: str, options: List[str]) -> None:
        """Debug helper to print response matching information"""
        print("\nDebug Response Matching:")
        print(f"User input: '{user_input}'")
        print(f"Normalized input: '{self._normalize_input(user_input)}'")

        matched = self._find_best_match(user_input, options)
        print(f"Matched option: {matched}")

        if matched is None:
            print("Tried matching against variations:")
            for base_word, variations in self._common_variations.items():
                if any(self._normalize_input(var) in self._normalize_input(user_input) for var in variations):
                    print(f"- Found variation match: {base_word} -> {variations}")


class EnhancedCareerCoachSystem(CareerCoachSystem):
    def __init__(self):
        super().__init__()
        self.rasa_client = SyncRasaClient()
        self.current_user_id = None
        self.conversation_state = {}
        self.interaction_handler = UserInteractionHandler(
            self.speech_system,
            self.speech_lock,
            self.engine
        )
        self._initialize_assessment_questions()
        self._initialize_career_details()

    def init_rasa_conversation(self, user_id: str):
        """Initialize a new RASA conversation"""
        self.current_user_id = user_id
        self.conversation_state[user_id] = {
            'slots': {},
            'active_form': None
        }

    def _initialize_assessment_questions(self):
        """Initialize the assessment questions with complete validation options"""
        self.assessment_questions = [
            {
                "id": "education",
                "text": "What's your highest level of education?",
                "options": ["Bachelor's", "Master's", "PhD"],
                "validation": ["bachelors", "masters", "phd", "bachelor", "master", "doctorate",
                               "Bachelor's", "Master's", "PhD", "bs", "ms", "doctorate"],
                "confirmation": "I see, {} education. Thank you.",
                "rasa_intent": "/provide_education"
            },
            {
                "id": "work_location",
                "text": "Do you prefer working remotely or in office?",
                "options": ["Remote", "In-Office", "Hybrid"],
                "validation": ["remote", "office", "hybrid", "remotely", "in-office", "from home",
                               "wfh", "onsite", "on-site", "Remote", "In-Office", "Hybrid"],
                "confirmation": "Got it, you prefer {} work.",
                "rasa_intent": "/provide_work_location"
            },
            {
                "id": "work_hours",
                "text": "What working hours do you prefer?",
                "options": ["Standard 9-5", "Flexible Hours", "Shift Work"],
                "validation": ["standard", "flexible", "shift", "9-5", "nine to five",
                               "regular", "flex", "variable", "shifts", "Standard 9-5",
                               "Flexible Hours", "Shift Work"],
                "confirmation": "I understand, you prefer {}.",
                "rasa_intent": "/provide_work_hours"
            },
            {
                "id": "stress_level",
                "text": "What level of work stress can you handle?",
                "options": ["Low", "Moderate", "High"],
                "validation": ["low", "moderate", "high", "minimal", "medium", "intense",
                               "light", "mid", "heavy", "Low", "Moderate", "High"],
                "confirmation": "Noted, {} stress level preference.",
                "rasa_intent": "/provide_stress_level"
            },
            {
                "id": "expected_salary",
                "text": "What's your expected salary range?",
                "options": ["Entry Level", "Mid Range", "Senior Level"],
                "validation": ["entry", "mid", "senior", "starting", "middle", "high",
                               "junior", "intermediate", "advanced", "Entry Level",
                               "Mid Range", "Senior Level"],
                "confirmation": "I understand, {} salary range.",
                "rasa_intent": "/provide_salary_expectation"
            },
            {
                "id": "team_size",
                "text": "What team size do you prefer working in?",
                "options": ["Small Team", "Medium Team", "Large Team"],
                "validation": ["small", "medium", "large", "tiny", "mid", "big",
                               "compact", "moderate", "huge", "Small Team", "Medium Team",
                               "Large Team"],
                "confirmation": "Understood, you prefer working in a {}.",
                "rasa_intent": "/provide_team_size"
            },
            {
                "id": "growth_speed",
                "text": "How fast do you want your career to grow?",
                "options": ["Steady Pace", "Moderate Growth", "Fast Track"],
                "validation": ["steady", "moderate", "fast", "slow", "medium", "quick",
                               "gradual", "balanced", "rapid", "Steady Pace",
                               "Moderate Growth", "Fast Track"],
                "confirmation": "Got it, you prefer {} for career growth.",
                "rasa_intent": "/provide_growth_speed"
            }
        ]

    def _initialize_career_details(self):
        """Initialize detailed career information for all career paths"""
        self.career_details = {
            'Software Developer': CareerDetails(
                title="Software Developer",
                description="Software developers create computer programs and applications. They need strong problem-solving skills and programming knowledge. They work on various platforms including web, mobile, and desktop applications.",
                skills=[
                    "Programming Languages (Python, Java, JavaScript)",
                    "Problem Solving",
                    "Analytical Thinking",
                    "Software Design Patterns",
                    "Version Control (Git)",
                    "Database Management",
                    "Testing and Debugging"
                ],
                education=[
                    "Bachelor's in Computer Science",
                    "Bachelor's in Software Engineering",
                    "Related Technical Degree",
                    "Coding Bootcamp (for entry level)"
                ],
                salary_range="$60,000 - $150,000"
            ),

            'Data Scientist': CareerDetails(
                title="Data Scientist",
                description="Data scientists analyze complex data sets to help businesses make better decisions. They combine statistics, mathematics, and programming to extract insights from data and build predictive models.",
                skills=[
                    "Statistics and Mathematics",
                    "Machine Learning",
                    "Programming (Python, R)",
                    "Data Visualization",
                    "Big Data Technologies",
                    "SQL and Database Management",
                    "Deep Learning"
                ],
                education=[
                    "Master's in Data Science",
                    "PhD in Statistics",
                    "Master's in Computer Science",
                    "Master's in Mathematics"
                ],
                salary_range="$70,000 - $160,000"
            ),

            'Project Manager': CareerDetails(
                title="Project Manager",
                description="Project managers oversee project planning, execution, and delivery. They coordinate teams, manage resources, and ensure projects are completed on time and within budget.",
                skills=[
                    "Leadership",
                    "Communication",
                    "Risk Management",
                    "Budgeting",
                    "Agile Methodologies",
                    "Stakeholder Management",
                    "Project Planning"
                ],
                education=[
                    "Bachelor's in Business Administration",
                    "Bachelor's in Management",
                    "PMP Certification",
                    "Agile Certifications"
                ],
                salary_range="$65,000 - $140,000"
            ),

            'UX Designer': CareerDetails(
                title="UX Designer",
                description="UX designers create user-friendly interfaces and experiences. They focus on understanding user needs and behaviors to design intuitive and effective digital products.",
                skills=[
                    "User Research",
                    "Wireframing",
                    "Prototyping",
                    "UI Design",
                    "User Testing",
                    "Design Tools (Figma, Sketch)",
                    "Information Architecture"
                ],
                education=[
                    "Bachelor's in Design",
                    "Bachelor's in Human-Computer Interaction",
                    "UX Design Certification",
                    "Related Design Degree"
                ],
                salary_range="$55,000 - $130,000"
            ),

            'Business Analyst': CareerDetails(
                title="Business Analyst",
                description="Business analysts bridge the gap between business and IT, analyzing processes and recommending improvements. They gather requirements and ensure solutions meet business needs.",
                skills=[
                    "Requirements Analysis",
                    "Process Modeling",
                    "Data Analysis",
                    "Documentation",
                    "SQL",
                    "Business Process Improvement",
                    "Stakeholder Management"
                ],
                education=[
                    "Bachelor's in Business Administration",
                    "Bachelor's in Information Systems",
                    "CBAP Certification",
                    "Related Business Degree"
                ],
                salary_range="$55,000 - $120,000"
            ),

            'DevOps Engineer': CareerDetails(
                title="DevOps Engineer",
                description="DevOps engineers manage the infrastructure and deployment pipelines. They automate processes, maintain systems, and ensure smooth operation of IT infrastructure.",
                skills=[
                    "CI/CD",
                    "Cloud Platforms (AWS, Azure)",
                    "Container Technologies",
                    "Infrastructure as Code",
                    "Scripting",
                    "Monitoring and Logging",
                    "Security Practices"
                ],
                education=[
                    "Bachelor's in Computer Science",
                    "Bachelor's in IT",
                    "Cloud Certifications",
                    "DevOps Certifications"
                ],
                salary_range="$75,000 - $155,000"
            ),

            'Product Manager': CareerDetails(
                title="Product Manager",
                description="Product managers define product vision and strategy. They work with various teams to ensure product development aligns with market needs and business goals.",
                skills=[
                    "Product Strategy",
                    "Market Research",
                    "Data Analysis",
                    "Agile Development",
                    "Stakeholder Management",
                    "User Experience",
                    "Business Strategy"
                ],
                education=[
                    "Bachelor's in Business",
                    "Bachelor's in Computer Science",
                    "MBA",
                    "Product Management Certification"
                ],
                salary_range="$70,000 - $160,000"
            ),

            'Systems Architect': CareerDetails(
                title="Systems Architect",
                description="Systems architects design and oversee the implementation of complex IT systems. They ensure systems are scalable, secure, and align with business requirements.",
                skills=[
                    "System Design",
                    "Enterprise Architecture",
                    "Cloud Architecture",
                    "Security Design",
                    "Technical Leadership",
                    "Solution Design",
                    "Integration Patterns"
                ],
                education=[
                    "Master's in Computer Science",
                    "Bachelor's in Software Engineering",
                    "Architecture Certifications",
                    "Cloud Architecture Certifications"
                ],
                salary_range="$90,000 - $180,000"
            ),

            'Data Engineer': CareerDetails(
                title="Data Engineer",
                description="Data engineers design and maintain data infrastructure. They create data pipelines, ensure data quality, and optimize data systems for performance.",
                skills=[
                    "Data Warehousing",
                    "ETL Processes",
                    "Big Data Technologies",
                    "Database Design",
                    "Data Modeling",
                    "Programming",
                    "Cloud Data Platforms"
                ],
                education=[
                    "Bachelor's in Computer Science",
                    "Bachelor's in Data Engineering",
                    "Data Engineering Certifications",
                    "Cloud Data Certifications"
                ],
                salary_range="$70,000 - $150,000"
            ),

            'Cybersecurity Analyst': CareerDetails(
                title="Cybersecurity Analyst",
                description="Cybersecurity analysts protect systems and networks from threats. They monitor security measures, investigate incidents, and implement security solutions.",
                skills=[
                    "Security Analysis",
                    "Threat Detection",
                    "Incident Response",
                    "Security Tools",
                    "Network Security",
                    "Risk Assessment",
                    "Security Compliance"
                ],
                education=[
                    "Bachelor's in Cybersecurity",
                    "Bachelor's in Computer Science",
                    "Security Certifications (CISSP, CEH)",
                    "Network Security Certifications"
                ],
                salary_range="$65,000 - $140,000"
            )
        }

    def start_interaction_with_user(self, user_info: Dict, greeting: str):
        """Start interaction with a detected user"""
        try:
            # Initialize RASA conversation
            self.init_rasa_conversation(user_info['name'])

            # Initial greeting with name
            self.execute_gesture('wave')
            personalized_greeting = (
                f"Hello {user_info['name']}! "
                f"{'Welcome back!' if user_info['user_type'] == 'team_member' else 'Nice to meet you!'} "
                "I'm your career coach!"
            )
            self.interaction_handler.speak_and_wait(personalized_greeting)

            # Select interaction mode
            mode_choice = self.interaction_handler.get_validated_input(
                "Would you like to proceed with voice interaction?",
                ["yes", "no"]
            )

            # Set interaction mode based on response
            self.interaction_mode = InteractionMode.VOICE if mode_choice.lower() == "yes" else InteractionMode.TEXT
            self.interaction_handler.interaction_mode = self.interaction_mode

            # Confirm mode selection
            mode_msg = "Voice interaction mode selected." if self.interaction_mode == InteractionMode.VOICE else "Text interaction mode selected."
            self.interaction_handler.speak_and_wait(mode_msg)

            # Conduct assessment
            responses = self.conduct_career_assessment(user_info)

            # Generate and present recommendations
            recommendations = self.generate_recommendations(responses)
            self.present_recommendations(recommendations)

            # Save session
            self.save_session(responses, recommendations)

            # Farewell
            farewell = self.rasa_client.send_message("/goodbye", self.current_user_id)
            self.interaction_handler.speak_and_wait(
                farewell or f"Thank you for using our career coaching service, {user_info['name']}! Good luck with your career journey!"
            )
            self.execute_gesture('wave')

        except Exception as e:
            print(f"Error in interaction: {e}")
            self.interaction_handler.speak_and_wait(
                "I apologize, but there seems to be an error. Let's try again later."
            )

    def conduct_career_assessment(self, user_info: Dict) -> Dict:
        """Conduct career assessment with improved error handling"""
        responses = {'user_name': user_info['name'], 'user_type': user_info['user_type']}

        print("\n" + "=" * 50)
        print("Career Assessment")
        print("=" * 50 + "\n")

        # Use the questions from self.assessment_questions instead of defining them here
        for question in self.assessment_questions:
            try:
                self.speak(question['text'])
                response = self.get_validated_input(
                    prompt=question['text'],
                    options=question['options'],
                    validation=question['validation']
                )
                responses[question['id']] = response

                # Use the confirmation template from the question if available
                confirmation = question.get('confirmation', "Got it, you selected: {}").format(response)
                self.speak(confirmation)
                print(f"\nConfirmed: {response}")

                self.nod()
                time.sleep(0.5)

            except Exception as e:
                print(f"Error processing question {question['id']}: {e}")
                # Use a default value if available
                if question['options']:
                    responses[question['id']] = question['options'][0]
                    print(f"Using default value: {question['options'][0]}")

        # Final confirmation
        self.speak("Thank you for providing all your preferences. I'll now analyze the best career options for you.")
        return responses

    def get_validated_input(self, prompt: str, options: List[str] = None, validation: List[str] = None) -> str:
        """Get and validate user input with improved error handling and matching"""
        max_retries = 3
        retries = 0

        def normalize_input(text: str) -> str:
            """Normalize input text by removing special characters and converting to lowercase"""
            return text.lower().strip().replace("'s", "").replace("'", "").replace("-", " ")

        # Pre-normalize options and validation lists for consistent matching
        normalized_options = {normalize_input(opt): opt for opt in (options or [])}
        normalized_validation = {normalize_input(val): val for val in (validation or [])}

        while retries < max_retries:
            try:
                # Clear section and display prompt
                print("\nUser Input Required:")
                print("-" * 40)
                print(f"{prompt}")

                if options:
                    print("\nAvailable options:")
                    for idx, option in enumerate(options, 1):
                        print(f"{idx}. {option}")

                # Get input with proper encoding handling
                try:
                    response = input("\nYour response: ").strip()
                except UnicodeDecodeError:
                    print("Invalid character encoding detected. Please try again.")
                    retries += 1
                    continue

                if not response:
                    print("Please provide a response.")
                    retries += 1
                    continue

                # Try to match by number first
                try:
                    choice_num = int(response)
                    if options and 1 <= choice_num <= len(options):
                        return options[choice_num - 1]
                except ValueError:
                    pass

                # Normalize the input for text matching
                normalized_response = normalize_input(response)

                # Check against normalized options and validation lists
                if normalized_response in normalized_options:
                    return normalized_options[normalized_response]
                elif normalized_response in normalized_validation:
                    # Map back to the corresponding original option
                    for opt in options:
                        if normalize_input(opt) == normalized_response:
                            return opt
                    return normalized_validation[normalized_response]

                # Special case handling for education levels
                if "bachelor" in normalized_response:
                    return "Bachelor's"
                elif "master" in normalized_response:
                    return "Master's"
                elif any(phd in normalized_response for phd in ["phd", "doctorate"]):
                    return "PhD"

                print("\nInvalid response. Please choose from the available options.")
                print("You can enter either the number or the text of your choice.")
                retries += 1

            except Exception as e:
                print(f"Error processing input: {str(e)}")
                retries += 1

        # Fallback after max retries
        if options:
            print("\nMaximum retries reached. Using the first option as default.")
            return options[0]
        raise ValueError("Failed to get valid input after multiple attempts.")

    def present_recommendations(self, recommendations: List[Dict]):
        """Present career recommendations with improved interaction"""
        try:
            # Clear section separation
            print("\n" + "=" * 50)
            print("Career Recommendations")
            print("=" * 50)

            # Get RASA introduction
            try:
                rasa_intro = self.rasa_client.send_message("show recommendations", self.current_user_id)
                self.interaction_handler.speak_and_wait(
                    rasa_intro or "Based on your preferences, here are your top career matches:"
                )
            except Exception as e:
                print(f"Warning: Failed to fetch RASA introduction: {e}")
                self.interaction_handler.speak_and_wait(
                    "Based on your preferences, here are your top career matches:"
                )

            # Present recommendations with proper spacing
            for i, rec in enumerate(recommendations, 1):
                try:
                    recommendation_text = f"{i}. {rec['title']} with {rec['match']}% match"
                    print(f"\n{recommendation_text}")
                    self.interaction_handler.speak_and_wait(recommendation_text)
                    time.sleep(0.5)
                except KeyError as e:
                    print(f"Error with recommendation format: {e}")

            # Save session
            try:
                self.save_session(self.current_responses, recommendations)
            except Exception as e:
                print(f"Warning: Could not save session: {e}")

            # Handle career details with improved interaction
            more_details = self.interaction_handler.get_validated_input(
                "Would you like to know more about any of these careers?",
                ["yes", "no"]
            )

            if more_details.lower() == "yes":
                career_options = [rec['title'] for rec in recommendations]
                career_choice = self.interaction_handler.get_validated_input(
                    "Which career would you like to know more about?",
                    career_options
                )

                if career_choice in career_options:
                    self.provide_career_details(career_choice)
                else:
                    self.interaction_handler.speak_and_wait(
                        "I'm sorry, I couldn't find that career in the recommendations."
                    )

        except Exception as e:
            print(f"Error in presenting recommendations: {e}")
            self.interaction_handler.speak_and_wait(
                "I'm sorry, there was an error presenting the recommendations."
            )

    def provide_career_details(self, career_title: str):
        """Provide career details with improved interaction"""
        try:
            rasa_response = self.rasa_client.send_message(
                f"tell me about {career_title}",
                self.current_user_id
            )

            if career_title in self.career_details:
                self.interaction_handler.speak_and_wait(
                    rasa_response or self.career_details[career_title]
                )
            else:
                self.interaction_handler.speak_and_wait(
                    rasa_response or
                    f"I'm sorry, I don't have detailed information about {career_title} at the moment."
                )
        except Exception as e:
            print(f"Error providing career details: {e}")
            self.interaction_handler.speak_and_wait(
                "I'm sorry, I had trouble retrieving the career details."
            )