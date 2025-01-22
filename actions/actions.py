from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
from bayesian_network import CareerBayesianNetwork
from difflib import get_close_matches
from dataclasses import dataclass

from typing import Any, Dict, Text, List
from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet, FollowupAction
from bayesian_network import CareerBayesianNetwork


class ValidateCareerForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_career_form"

    def validate_education_level(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        education = slot_value.lower()
        if any(edu in education for edu in ["bachelor", "master", "phd", "doctorate"]):
            return {"education_level": education}
        else:
            dispatcher.utter_message(text="Please provide a valid education level (Bachelor's, Master's, or PhD)")
            return {"education_level": None}

    def validate_location_preference(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        location = slot_value.lower()
        if any(loc in location for loc in ["remote", "office", "hybrid"]):
            return {"location_preference": location}
        else:
            dispatcher.utter_message(text="Please choose remote, in-office, or hybrid")
            return {"location_preference": None}

    def validate_hours_preference(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        hours = slot_value.lower()
        if any(h in hours for h in ["9-5", "nine", "standard", "flexible", "shift"]):
            return {"hours_preference": hours}
        else:
            dispatcher.utter_message(text="Please choose Standard 9-5, Flexible Hours, or Shift Work")
            return {"hours_preference": None}

    def validate_stress_level(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        stress = slot_value.lower()
        if any(s in stress for s in ["low", "moderate", "high"]):
            return {"stress_level": stress}
        else:
            dispatcher.utter_message(text="Please choose Low, Moderate, or High")
            return {"stress_level": None}

    def validate_salary_range(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        salary = slot_value.lower()
        if any(s in salary for s in ["entry", "junior", "mid", "senior"]):
            return {"salary_range": salary}
        else:
            dispatcher.utter_message(text="Please choose Entry Level, Mid Range, or Senior Level")
            return {"salary_range": None}

    def validate_team_size(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        team = slot_value.lower()
        if any(t in team for t in ["small", "medium", "large", "big"]):
            return {"team_size": team}
        else:
            dispatcher.utter_message(text="Please choose Small Team, Medium Team, or Large Team")
            return {"team_size": None}

    def validate_growth_speed(
            self,
            slot_value: Any,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: DomainDict,
    ) -> Dict[Text, Any]:
        growth = slot_value.lower()
        if any(g in growth for g in ["steady", "moderate", "fast", "rapid"]):
            return {"growth_speed": growth}
        else:
            dispatcher.utter_message(text="Please choose Steady Pace, Moderate Growth, or Fast Track")
            return {"growth_speed": None}


class ActionProvideCareerMatches(Action):
    def name(self) -> Text:
        return "action_provide_careers"

    def _normalize_preferences(self, preferences: Dict[str, str]) -> Dict[str, str]:
        """Normalize user preferences to match Bayesian network expectations"""
        normalized = {}

        # Education mapping
        if preferences.get("education_level"):
            edu = preferences["education_level"].lower()
            if "bachelor" in edu:
                normalized["education"] = "Bachelor's"
            elif "master" in edu:
                normalized["education"] = "Master's"
            elif "phd" in edu or "doctorate" in edu:
                normalized["education"] = "PhD"

        # Work location mapping
        if preferences.get("location_preference"):
            loc = preferences["location_preference"].lower()
            if "remote" in loc:
                normalized["work_location"] = "Remote"
            elif "office" in loc:
                normalized["work_location"] = "In-Office"
            elif "hybrid" in loc:
                normalized["work_location"] = "Hybrid"

        # Working hours mapping
        if preferences.get("hours_preference"):
            hours = preferences["hours_preference"].lower()
            if any(h in hours for h in ["9-5", "nine", "standard"]):
                normalized["work_hours"] = "Standard 9-5"
            elif "flexible" in hours:
                normalized["work_hours"] = "Flexible Hours"
            elif "shift" in hours:
                normalized["work_hours"] = "Shift Work"

        # Stress level mapping
        if preferences.get("stress_level"):
            stress = preferences["stress_level"].lower()
            normalized["stress_level"] = stress.capitalize()

        # Salary range mapping
        if preferences.get("salary_range"):
            salary = preferences["salary_range"].lower()
            if "entry" in salary or "junior" in salary:
                normalized["expected_salary"] = "Entry Level"
            elif "mid" in salary:
                normalized["expected_salary"] = "Mid Range"
            elif "senior" in salary:
                normalized["expected_salary"] = "Senior Level"

        # Team size mapping
        if preferences.get("team_size"):
            team = preferences["team_size"].lower()
            if "small" in team:
                normalized["team_size"] = "Small Team"
            elif "medium" in team:
                normalized["team_size"] = "Medium Team"
            elif "large" in team or "big" in team:
                normalized["team_size"] = "Large Team"

        # Growth speed mapping
        if preferences.get("growth_speed"):
            growth = preferences["growth_speed"].lower()
            if "steady" in growth or "slow" in growth:
                normalized["growth_speed"] = "Steady Pace"
            elif "moderate" in growth:
                normalized["growth_speed"] = "Moderate Growth"
            elif "fast" in growth or "rapid" in growth:
                normalized["growth_speed"] = "Fast Track"

        return normalized

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get all preferences from slots
        raw_preferences = {
            "education_level": tracker.get_slot("education_level"),
            "location_preference": tracker.get_slot("location_preference"),
            "hours_preference": tracker.get_slot("hours_preference"),
            "stress_level": tracker.get_slot("stress_level"),
            "salary_range": tracker.get_slot("salary_range"),
            "team_size": tracker.get_slot("team_size"),
            "growth_speed": tracker.get_slot("growth_speed")
        }

        # Normalize preferences for Bayesian network
        normalized_preferences = self._normalize_preferences(raw_preferences)

        # Get recommendations from Bayesian network
        bn = CareerBayesianNetwork.get_instance()
        recommendations = bn.get_recommendations(normalized_preferences, top_n=3)

        # Create response with preferences summary
        response = "Based on your preferences:\n"
        for pref, value in raw_preferences.items():
            if value:  # Only show preferences that were provided
                response += f"{pref.replace('_', ' ').title()}: {value}\n"

        response += "\nHere are your top career matches:\n"
        for i, career in enumerate(recommendations, 1):
            response += f"{i}. {career['title']} with {career['match']}% match\n"

        response += "\nWould you like to know more about any of these careers?"

        # Send response to user
        dispatcher.utter_message(text=response)

        # Store recommendations for later use
        return [SlotSet("career_recommendations", recommendations)]


class ActionProvideCareerDetails(Action):
    def name(self) -> Text:
        return "action_provide_career_details"

    def extract_section(self, career_text: str, section_name: str) -> str:
        """Extract a specific section from the career description"""
        try:
            sections = career_text.split('\n\n')
            for section in sections:
                if section_name in section:
                    return section.strip()
            return None
        except Exception:
            return None

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        last_message = tracker.latest_message.get('text', '').lower()
        last_action = tracker.latest_action_name if tracker.latest_action_name else None

        # Dictionary mapping keywords to section names
        section_keywords = {
            'responsibility': 'Key Responsibilities',
            'task': 'Key Responsibilities',
            'do': 'Key Responsibilities',
            'skill': 'Required Skills',
            'requirement': 'Required Skills',
            'need': 'Required Skills',
            'environment': 'Work Environment',
            'workplace': 'Work Environment',
            'setting': 'Work Environment',
            'salary': 'Salary Range',
            'pay': 'Salary Range',
            'money': 'Salary Range',
            'growth': 'Growth Opportunities',
            'career path': 'Growth Opportunities',
            'advance': 'Growth Opportunities'
        }

        # Handle generic responses based on context
        if last_message in ['yes', 'yeah', 'sure', 'okay', 'ok', 'yep', 'y']:
            if tracker.active_loop:
                return []

            if last_action == 'action_provide_careers':
                # After showing career matches
                dispatcher.utter_message(text="Which career would you like to know more about?")
                return []
            else:
                # During career exploration
                dispatcher.utter_message(text="What would you like to know? You can ask about:")
                dispatcher.utter_message(
                    text="1. Salary and compensation\n2. Required skills and qualifications\n3. Work environment\n4. Career growth opportunities\n5. Day-to-day responsibilities")
                return []

        elif last_message in ['no', 'nope', 'nah', 'n']:
            if tracker.active_loop:
                return []
            dispatcher.utter_message(
                text="Thank you for using our career coaching service! Good luck with your career journey!")
            return []

        elif last_message in ['k', 'kk', 'hmm', 'um', 'good']:
            if tracker.active_loop:
                return []
            dispatcher.utter_message(text="Would you like to know anything specific about any of these careers?")
            return []
        # Comprehensive career details dictionary
        career_details = {
            "software developer": """A Software Developer creates computer programs and applications. Here's what you need to know:

Key Responsibilities:
- Writing and testing code for new programs
- Maintaining and improving existing software
- Collaborating with development teams
- Debugging and troubleshooting code
- Implementing software development best practices

Required Skills:
- Programming languages (Python, Java, JavaScript, etc.)
- Version control systems (Git)
- Problem-solving abilities
- Database management
- Software testing

Work Environment:
- Typically work in office or hybrid settings
- Often involves team collaboration
- Project-based work structure
- Regular code reviews and meetings
- Continuous learning environment

Salary Range:
- Entry Level: $60,000-$80,000
- Mid-Level: $80,000-$120,000
- Senior Level: $120,000-$200,000+

Growth Opportunities:
- Senior Software Developer
- Technical Lead
- Software Architect
- Development Manager
- Chief Technology Officer (CTO)""",

            "data scientist": """A Data Scientist analyzes complex data to help businesses make better decisions. Here's what you need to know:

Key Responsibilities:
- Collecting and analyzing large datasets
- Building predictive models
- Creating data visualizations
- Developing machine learning algorithms
- Communicating findings to stakeholders

Required Skills:
- Python/R programming
- Statistical analysis
- Machine learning
- SQL and database management
- Data visualization tools
- Communication skills

Work Environment:
- Mix of independent and team work
- Project-based assignments
- Regular stakeholder presentations
- Research and development focus
- Cross-functional collaboration

Salary Range:
- Entry Level: $70,000-$90,000
- Mid-Level: $90,000-$130,000
- Senior Level: $130,000-$200,000+

Growth Opportunities:
- Senior Data Scientist
- Lead Data Scientist
- AI Research Scientist
- Analytics Manager
- Chief Data Officer (CDO)""",

            "project manager": """A Project Manager oversees project planning, execution, and delivery. Here's what you need to know:

Key Responsibilities:
- Planning and defining project scope
- Resource allocation and management
- Risk assessment and mitigation
- Budget management
- Team leadership and coordination
- Stakeholder communication

Required Skills:
- Project management methodologies
- Leadership and team management
- Budget planning and control
- Risk management
- Communication and presentation
- Problem-solving abilities

Work Environment:
- High interaction with teams
- Regular stakeholder meetings
- Deadline-driven environment
- Multiple project coordination
- Cross-departmental collaboration

Salary Range:
- Entry Level: $55,000-$75,000
- Mid-Level: $75,000-$110,000
- Senior Level: $110,000-$150,000+

Growth Opportunities:
- Senior Project Manager
- Program Manager
- Portfolio Manager
- Director of Project Management
- Chief Operating Officer (COO)""",

            "ux designer": """A UX Designer creates user-friendly digital experiences. Here's what you need to know:

Key Responsibilities:
- User research and testing
- Creating wireframes and prototypes
- Designing user interfaces
- Usability testing
- Information architecture
- User journey mapping

Required Skills:
- Design tools (Figma, Sketch, Adobe XD)
- User research methods
- Prototyping
- Information architecture
- Visual design principles
- HTML/CSS basics

Work Environment:
- Creative atmosphere
- User-centered focus
- Collaborative design teams
- Regular user testing sessions
- Iterative design process

Salary Range:
- Entry Level: $55,000-$75,000
- Mid-Level: $75,000-$100,000
- Senior Level: $100,000-$150,000+

Growth Opportunities:
- Senior UX Designer
- UX Lead
- Design Manager
- UX Director
- Chief Design Officer (CDO)""",

            "business analyst": """A Business Analyst bridges the gap between business needs and technology solutions. Here's what you need to know:

Key Responsibilities:
- Requirements gathering and analysis
- Process modeling and optimization
- Stakeholder interviews
- Documentation creation
- Solution testing and validation
- Change management support

Required Skills:
- Business process modeling
- Requirements analysis
- Data analysis
- Technical documentation
- Communication skills
- Problem-solving abilities

Work Environment:
- Mix of business and IT interaction
- Regular stakeholder meetings
- Process improvement focus
- Documentation-heavy role
- Cross-functional collaboration

Salary Range:
- Entry Level: $50,000-$70,000
- Mid-Level: $70,000-$95,000
- Senior Level: $95,000-$130,000+

Growth Opportunities:
- Senior Business Analyst
- Lead Business Analyst
- Business Architecture Manager
- Director of Business Analysis
- Chief Business Analyst""",

            "devops engineer": """A DevOps Engineer manages infrastructure and deployment processes. Here's what you need to know:

Key Responsibilities:
- CI/CD pipeline management
- Infrastructure automation
- System monitoring and maintenance
- Security implementation
- Performance optimization
- Tool development and integration

Required Skills:
- Cloud platforms (AWS, Azure, GCP)
- Infrastructure as Code
- Containerization (Docker, Kubernetes)
- Scripting languages
- Monitoring tools
- Security practices

Work Environment:
- Fast-paced environment
- On-call rotations common
- High automation focus
- Incident response handling
- Continuous improvement culture

Salary Range:
- Entry Level: $65,000-$85,000
- Mid-Level: $85,000-$125,000
- Senior Level: $125,000-$180,000+

Growth Opportunities:
- Senior DevOps Engineer
- DevOps Architect
- Site Reliability Engineer
- Infrastructure Manager
- Chief Technology Officer (CTO)"""
        }

        # Find which career the user is asking about
        career_found = None
        for career in career_details.keys():
            if career in last_message:
                career_found = career
                break

        if not career_found:
            # Only show available careers if we're not in a form and haven't found a career
            if not tracker.active_loop:
                available_careers = ", ".join(career_details.keys()).title()
                dispatcher.utter_message(
                    text=f"Please specify which career you'd like to know more about: {available_careers}")
            return []

        # Determine if user is asking about a specific aspect
        section_to_show = None
        for keyword, section in section_keywords.items():
            if keyword in last_message:
                section_to_show = section
                break

        if section_to_show:
            # Extract and show specific section
            section_content = self.extract_section(career_details[career_found], section_to_show)
            if section_content:
                dispatcher.utter_message(text=section_content)
                dispatcher.utter_message(text="Would you like to know about any other aspects of this career?")
            else:
                dispatcher.utter_message(text=career_details[career_found])
        else:
            # Show full career details if no specific aspect is asked
            dispatcher.utter_message(text=career_details[career_found])
            dispatcher.utter_message(text="Is there any specific aspect of this career you'd like to know more about?")

        return []


class ActionAnswerCareerQuestion(Action):
    def name(self) -> Text:
        return "action_answer_career_question"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        career = tracker.get_slot("career_name")
        message = tracker.latest_message.get('text', '').lower()

        career_info = {
            "Software Developer": {
                "salary": "Entry Level: $60,000-$80,000\nMid-Level: $80,000-$120,000\nSenior Level: $120,000-$200,000+",
                "skills": "Programming languages (Python, Java, JavaScript), version control systems (Git), problem-solving abilities, database management, software testing",
                "education": "Bachelor's degree in Computer Science or related field",
                "environment": "Office or hybrid settings, team collaboration, project-based work structure, regular code reviews and meetings, continuous learning environment",
                "growth": "Senior Software Developer, Technical Lead, Software Architect, Development Manager, Chief Technology Officer (CTO)"
            },
            "Data Scientist": {
                "salary": "Entry Level: $70,000-$90,000\nMid-Level: $90,000-$130,000\nSenior Level: $130,000-$200,000+",
                "skills": "Python/R programming, statistical analysis, machine learning, SQL and database management, data visualization tools, communication skills",
                "education": "Master's degree in Statistics, Mathematics, Computer Science or related field",
                "environment": "Mix of independent and team work, project-based assignments, regular stakeholder presentations, research and development focus, cross-functional collaboration",
                "growth": "Senior Data Scientist, Lead Data Scientist, AI Research Scientist, Analytics Manager, Chief Data Officer (CDO)"
            },
            "Project Manager": {
                "salary": "Entry Level: $55,000-$75,000\nMid-Level: $75,000-$110,000\nSenior Level: $110,000-$150,000+",
                "skills": "Project management methodologies, leadership and team management, budget planning and control, risk management, communication and presentation, problem-solving abilities",
                "education": "Bachelor's degree in Business, Management or related field, PMP certification preferred",
                "environment": "High interaction with teams, regular stakeholder meetings, deadline-driven environment, multiple project coordination, cross-departmental collaboration",
                "growth": "Senior Project Manager, Program Manager, Portfolio Manager, Director of Project Management, Chief Operating Officer (COO)"
            },
            "UX Designer": {
                "salary": "Entry Level: $55,000-$75,000\nMid-Level: $75,000-$100,000\nSenior Level: $100,000-$150,000+",
                "skills": "Design tools (Figma, Sketch, Adobe XD), user research methods, prototyping, information architecture, visual design principles, HTML/CSS basics",
                "education": "Bachelor's degree in Design, Human-Computer Interaction, or related field",
                "environment": "Creative atmosphere, user-centered focus, collaborative design teams, regular user testing sessions, iterative design process",
                "growth": "Senior UX Designer, UX Lead, Design Manager, UX Director, Chief Design Officer (CDO)"
            },
            "Business Analyst": {
                "salary": "Entry Level: $50,000-$70,000\nMid-Level: $70,000-$95,000\nSenior Level: $95,000-$130,000+",
                "skills": "Business process modeling, requirements analysis, data analysis, technical documentation, communication skills, problem-solving abilities",
                "education": "Bachelor's degree in Business, Information Systems, or related field",
                "environment": "Mix of business and IT interaction, regular stakeholder meetings, process improvement focus, documentation-heavy role, cross-functional collaboration",
                "growth": "Senior Business Analyst, Lead Business Analyst, Business Architecture Manager, Director of Business Analysis, Chief Business Analyst"
            },
            "DevOps Engineer": {
                "salary": "Entry Level: $65,000-$85,000\nMid-Level: $85,000-$125,000\nSenior Level: $125,000-$180,000+",
                "skills": "Cloud platforms (AWS, Azure, GCP), Infrastructure as Code, containerization (Docker, Kubernetes), scripting languages, monitoring tools, security practices",
                "education": "Bachelor's degree in Computer Science, System Engineering or related field",
                "environment": "Fast-paced environment, on-call rotations common, high automation focus, incident response handling, continuous improvement culture",
                "growth": "Senior DevOps Engineer, DevOps Architect, Site Reliability Engineer, Infrastructure Manager, Chief Technology Officer (CTO)"
            }
        }

        if career in career_info:
            if "salary" in message:
                response = f"For a {career}, the salary ranges are:\n{career_info[career]['salary']}"
            elif any(word in message for word in ["skills", "experience", "need"]):
                response = f"Key skills needed for {career}:\n{career_info[career]['skills']}"
            elif "education" in message:
                response = f"Education requirements for {career}:\n{career_info[career]['education']}"
            elif any(word in message for word in ["environment", "work", "hours"]):
                response = f"Work environment for {career}:\n{career_info[career]['environment']}"
            elif any(word in message for word in ["growth", "path", "advance"]):
                response = f"Career growth opportunities for {career}:\n{career_info[career]['growth']}"
            else:
                response = "You can ask about salary, skills, education, work environment, or growth opportunities."

            dispatcher.utter_message(text=response)
        else:
            dispatcher.utter_message(
                text="I don't have information about that career. Please ask about the careers from available options!.")

        return []