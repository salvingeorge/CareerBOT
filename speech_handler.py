import speech_recognition as speech_rec
import speech_recognition
print(dir(speech_recognition))
import threading
import time
import pyttsx3
from difflib import get_close_matches


class SpeechSystem:
    def __init__(self):
        # Initialize speech recognition
        self.recognizer = speech_rec.Recognizer()  # Fixed reference
        self.microphone = speech_rec.Microphone()  # Fixed reference

        # Initialize text-to-speech
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.speech_lock = threading.Lock()

        # Word variations dictionary
        self.word_variations = {
            "bachelor's": ["bachelors", "bachelor", "bachelor's", "bachelors degree", "bachelor degree"],
            "master's": ["masters", "master", "master's", "masters degree", "master degree"],
            "phd": ["phd", "doctorate", "doctoral", "ph.d", "ph.d.", "ph d"],
            "remote": ["remote", "work from home", "wfh", "remotely", "remote work"],
            "in-office": ["in office", "office", "in-office", "onsite", "on-site", "on site"],
            "hybrid": ["hybrid", "mixed", "flexible location", "partially remote"],
            "small team": ["small", "small team", "smaller team", "small group"],
            "medium team": ["medium", "medium team", "mid size", "mid-size"],
            "large team": ["large", "large team", "big team", "bigger team"],
            "entry level": ["entry", "entry-level", "entry level", "beginner", "starting"],
            "mid range": ["mid", "middle", "mid-range", "mid range", "intermediate"],
            "senior level": ["senior", "senior-level", "senior level", "high", "advanced"],
            "steady pace": ["steady", "steady pace", "slow", "slower", "gradual"],
            "moderate growth": ["moderate", "moderate growth", "medium pace", "medium growth"],
            "fast track": ["fast", "fast track", "fast-track", "rapid", "quick"],
            "low": ["low", "low stress", "minimal", "minimal stress", "light", "easy", "relaxed", "comfortable"],
            "moderate": ["moderate", "medium", "medium stress", "average", "normal", "intermediate", "balanced"],
            "high": ["high", "high stress", "intense", "challenging", "demanding", "heavy", "lots", "significant"]
        }

        # Adjust for ambient noise
        with self.microphone as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source)

    import speech_recognition as speech_rec
    import threading
    import time
    import pyttsx3
    from difflib import get_close_matches

    def get_standard_form(self, input_text: str, options: list) -> str:
        """Convert various forms of input to standard option format"""
        input_text = input_text.lower().strip()

        # First try exact match
        if input_text in [opt.lower() for opt in options]:
            return next(opt for opt in options if opt.lower() == input_text)

        # Check word variations
        for standard_form, variations in self.word_variations.items():
            if input_text in variations:
                matching_option = next((opt for opt in options if standard_form.lower() in opt.lower()), None)
                if matching_option:
                    return matching_option

        # Use difflib for fuzzy matching
        all_variations = []
        for opt in options:
            opt_lower = opt.lower()
            # Add the option itself and its variations
            all_variations.extend([(v, opt) for v in self.word_variations.get(opt_lower, [opt_lower])])

        # Flatten variations list
        variation_texts = [v[0] for v in all_variations]

        # Get close matches
        matches = get_close_matches(input_text, variation_texts, n=1, cutoff=0.6)
        if matches:
            # Find the original option for this match
            for variation, original in all_variations:
                if variation == matches[0]:
                    return original

        return None

    def speak(self, text: str, start_time: float = 0.0):
        """Make Pepper speak using pyttsx3"""
        time.sleep(start_time)
        print(f'Speaking speechhandler: {text}')

        with self.speech_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Speech error: {e}")

    def listen_for_speech(self, timeout=5):
        """Listen for speech and return the recognized text"""
        try:
            with self.microphone as source:
                print("\nListening...")
                audio = self.recognizer.listen(source, timeout=timeout)
                print("Processing speech...")

                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    return text.lower()
                except speech_rec.UnknownValueError:
                    print("Could not understand audio")
                    return None
                except speech_rec.RequestError as e:
                    print(f"Could not request results; {e}")
                    return None
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return None

    def get_voice_input(self, prompt=None, timeout=5, max_retries=3):
        """Get voice input with retries and fallback to text input"""
        if prompt:
            print(prompt)
            self.speak(prompt)

        for attempt in range(max_retries):
            result = self.listen_for_speech(timeout)
            if result:
                return result
            print(f"Didn't catch that. {max_retries - attempt - 1} attempts remaining.")
            self.speak("I didn't catch that. Please try again.")

        print("Falling back to text input.")
        self.speak("Please type your response instead.")
        return input("Your response: ")

    def confirm_voice_input(self, text, timeout=5):
        """Get confirmation for voice input"""
        print(f"I heard: {text}")
        self.speak(f"I heard: {text}. Is this correct?")

        confirmation = self.get_voice_input(timeout=timeout)
        return confirmation and ('yes' in confirmation.lower() or 'correct' in confirmation.lower())

    def match_voice_to_options(self, voice_input: str, options: list) -> int:
        """Match voice input to available options and return the index"""
        if not voice_input:
            return -1

        # Convert to lowercase for matching
        voice_input = voice_input.lower()

        # Try to match the exact number if spoken
        number_words = {
            'one': 0, 'first': 0, '1': 0,
            'two': 1, 'second': 1, '2': 1,
            'three': 2, 'third': 2, '3': 2,
            'four': 3, 'fourth': 3, '4': 3,
            'five': 4, 'fifth': 4, '5': 4
        }

        for word, index in number_words.items():
            if word in voice_input and index < len(options):
                return index

        # Try to match the option content
        for i, option in enumerate(options):
            option_lower = option.lower()
            if option_lower in voice_input or voice_input in option_lower:
                return i

        return -1

    def handle_question(self, question, options):
        """Process voice input for a question with multiple options"""
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            # Present question and options
            self.speak(question)
            print("\nOptions:")
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
                self.speak(f"Option {i}: {option}")
                time.sleep(0.5)

            self.speak("Please state your choice, or say 'repeat' to hear the options again.")
            voice_input = self.get_voice_input(timeout=7)

            if not voice_input:
                self.speak("I didn't hear anything. Let's try again.")
                attempt += 1
                continue

            # Handle repeat request
            if 'repeat' in voice_input.lower():
                continue

            # First try matching using our word variations system
            matched_option = self.get_standard_form(voice_input, options)

            # If that fails, try matching by index/position
            if not matched_option:
                choice_index = self.match_voice_to_options(voice_input, options)
                if choice_index >= 0:
                    matched_option = options[choice_index]

            if matched_option:
                # Confirm the selection
                self.speak(f"I heard you say {matched_option}. Is this correct?")
                confirmation = self.get_voice_input(timeout=5)

                if confirmation and any(word in confirmation.lower()
                                        for word in ['yes', 'yeah', 'correct', 'right', 'yep', 'sure']):
                    return matched_option
                else:
                    self.speak("Ok, let's try again.")
            else:
                self.speak("I didn't understand your choice. Let's try again.")

            attempt += 1

        # If we've exceeded max attempts, fall back to the first option
        self.speak(f"I'm having trouble understanding. I'll select {options[0]} for now.")
        return options[0]


def test_speech_system():
    """Test the speech recognition system"""
    try:
        speech_system = SpeechSystem()

        # Test basic interaction
        speech_system.speak("Hello! Let's test the speech recognition system.")
        time.sleep(1)

        # Test question handling
        question = "What's your favorite color?"
        options = ["Red", "Blue", "Green", "Yellow"]

        result = speech_system.handle_question(question, options)
        print(f"Selected option: {result}")

    except Exception as e:
        print(f"Error in test: {e}")


if __name__ == "__main__":
    test_speech_system()