import requests
import json
import time


def chat_with_rasa():
    """Interactive chat with RASA server through REST API"""
    base_url = "http://localhost:5005/webhooks/rest/webhook"
    sender_id = f"user_{int(time.time())}"  # Unique ID for each session

    print("Starting chat with RASA Career Coach...")
    print("Type 'quit' to end the conversation")
    print("-" * 50)

    while True:
        # Get user input
        user_message = input("\nYou: ").strip()

        if user_message.lower() in ['quit', 'exit', 'bye']:
            print("\nEnding chat session...")
            break

        # Prepare the payload
        payload = {
            "sender": sender_id,
            "message": user_message
        }

        try:
            # Send request to RASA
            response = requests.post(base_url, json=payload)

            if response.status_code == 200:
                responses = response.json()
                if responses:
                    print("\nBot:")
                    for resp in responses:
                        if 'text' in resp:
                            print(f"{resp['text']}")
                        if 'custom' in resp:
                            print(f"[Custom Action: {resp['custom']}]")
                else:
                    print("\nBot: I'm processing your request...")
            else:
                print(f"\nError: Received status code {response.status_code}")
                print(f"Response: {response.text}")

        except requests.exceptions.ConnectionError:
            print("\nError: Cannot connect to RASA server. Is it running?")
            print("Start the server with: rasa run -m models --enable-api --cors \"*\"")
            break
        except requests.exceptions.RequestException as e:
            print(f"\nError communicating with RASA server: {e}")
            break


def run_test_conversation():
    """Run a predefined test conversation"""
    base_url = "http://localhost:5005/webhooks/rest/webhook"
    sender_id = f"test_user_{int(time.time())}"

    test_messages = [
        "Hello",  # Greeting
        "I want career advice",  # Start assessment
        "I have a Master's degree",  # Education
        "I prefer remote work",  # Work location
        "Flexible hours",  # Working hours
        "Moderate stress",  # Stress level
        "Mid range salary",  # Salary
        "Medium team size",  # Team size
        "Fast track growth",  # Growth preference
        "Tell me more about Software Developer",  # Ask for details
        "goodbye"  # End conversation
    ]

    print("Running test conversation...")
    print("-" * 50)

    for message in test_messages:
        payload = {
            "sender": sender_id,
            "message": message
        }

        print(f"\nUser: {message}")

        try:
            response = requests.post(base_url, json=payload)

            if response.status_code == 200:
                responses = response.json()
                if responses:
                    print("Bot:")
                    for resp in responses:
                        if 'text' in resp:
                            print(f"{resp['text']}")
                else:
                    print("Bot: No response received")
            else:
                print(f"Error: Status code {response.status_code}")

        except requests.exceptions.ConnectionError:
            print("\nError: Cannot connect to RASA server. Is it running?")
            print("Start the server with: rasa run -m models --enable-api --cors \"*\"")
            return
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return

        time.sleep(1)  # Wait between messages


if __name__ == "__main__":
    print("Choose mode:")
    print("1. Interactive chat")
    print("2. Run test conversation")

    choice = input("\nEnter your choice (1 or 2): ").strip()

    try:
        if choice == "1":
            chat_with_rasa()
        elif choice == "2":
            run_test_conversation()
        else:
            print("Invalid choice. Please enter 1 or 2.")
    except KeyboardInterrupt:
        print("\nChat session interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")