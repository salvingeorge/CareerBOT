from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from bayesian_network import CareerBayesianNetwork


class ActionGenerateRecommendations(Action):
    def name(self) -> Text:
        return "action_generate_recommendations"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # Get preferences from slots
        preferences = {
            'education': tracker.get_slot('education'),
            'work_location': tracker.get_slot('work_location'),
            'work_hours': tracker.get_slot('work_hours'),
            'stress_level': tracker.get_slot('stress_level'),
            'expected_salary': tracker.get_slot('expected_salary'),
            'team_size': tracker.get_slot('team_size'),
            'growth_speed': tracker.get_slot('growth_speed')
        }

        # Get recommendations from Bayesian network
        network = CareerBayesianNetwork.get_instance()
        recommendations = network.get_recommendations(preferences)

        # Format and send recommendations
        response = "Based on your preferences, here are your top career matches:\n\n"
        for i, rec in enumerate(recommendations, 1):
            response += f"{i}. {rec['title']} with {rec['match']}% match\n"

        response += "\nWould you like to know more about any of these careers?"
        dispatcher.utter_message(text=response)

        return []


class ActionProvideCareerDetails(Action):
    def name(self) -> Text:
        return "action_provide_career_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Career details dictionary
        career_details = {
            'Software Developer': "Software developers create computer programs and applications. They need strong problem-solving skills and programming knowledge.",
            'Data Scientist': "Data scientists analyze complex data sets to help businesses make better decisions. Strong mathematics and statistics skills are essential.",
            'Project Manager': "Project managers oversee project planning, execution, and delivery. They need excellent organizational and leadership skills.",
            'UX Designer': "UX designers create user-friendly interfaces and experiences. They combine creativity with user research skills.",
            'Business Analyst': "Business analysts bridge the gap between business and IT, analyzing processes and recommending improvements.",
            'DevOps Engineer': "DevOps engineers manage the infrastructure and deployment pipelines. They need strong automation and systems knowledge."
        }

        # Get the requested career from the latest message
        for entity in tracker.latest_message.get('entities', []):
            if entity['entity'] == 'career_title':
                career = entity['value']
                if career in career_details:
                    dispatcher.utter_message(text=career_details[career])
                    return []

        # If no specific career was found or matched
        dispatcher.utter_message(
            text="I'm sorry, I couldn't find detailed information about that career. Could you please specify which career from the recommendations you'd like to know more about?")
        return []