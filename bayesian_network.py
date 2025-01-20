import pyAgrum as gum
import numpy as np
from typing import Dict, List, Optional
import threading

class CareerBayesianNetwork:
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.bn = gum.BayesNet('Career Recommendation')
        self._create_network()

    def _create_network(self):
        """Create simplified Bayesian network with valid CPTs"""
        # Level 1: Education
        self.education = self.bn.add(gum.LabelizedVariable('education', 'Education',
                                                           ["Bachelor's", "Master's", "PhD"]))

        # Level 2: Work Preferences
        self.work_location = self.bn.add(gum.LabelizedVariable('work_location', 'Work Location',
                                                               ['Remote', 'In-Office', 'Hybrid']))
        self.work_hours = self.bn.add(gum.LabelizedVariable('work_hours', 'Working Hours',
                                                            ['Standard 9-5', 'Flexible Hours', 'Shift Work']))
        self.stress_level = self.bn.add(gum.LabelizedVariable('stress_level', 'Stress Level',
                                                              ['Low', 'Moderate', 'High']))

        # Level 3: Personal Preferences
        self.team_size = self.bn.add(gum.LabelizedVariable('team_size', 'Team Size',
                                                           ['Small Team', 'Medium Team', 'Large Team']))
        self.growth_speed = self.bn.add(gum.LabelizedVariable('growth_speed', 'Growth Speed',
                                                              ['Steady Pace', 'Moderate Growth', 'Fast Track']))
        self.expected_salary = self.bn.add(gum.LabelizedVariable('expected_salary',
                                                                 'Expected Salary',
                                                                 ['Entry Level', 'Mid Range', 'Senior Level']))

        # Career Suggestions
        self.career = self.bn.add(gum.LabelizedVariable('career', 'Career Options',
                                                        ['Software Developer', 'Data Scientist', 'Project Manager',
                                                         'UX Designer', 'Business Analyst', 'DevOps Engineer']))

        # Add arcs
        self.bn.addArc(self.education, self.career)
        self.bn.addArc(self.work_location, self.career)
        self.bn.addArc(self.work_hours, self.career)
        self.bn.addArc(self.stress_level, self.career)
        self.bn.addArc(self.team_size, self.career)
        self.bn.addArc(self.growth_speed, self.career)
        self.bn.addArc(self.expected_salary, self.career)

        self._set_probability_tables()

    def _set_probability_tables(self):
        """Set valid conditional probability tables"""
        # Prior probabilities for each node
        self.bn.cpt(self.education).fillWith([0.4, 0.4, 0.2])
        self.bn.cpt(self.work_location).fillWith([0.3, 0.3, 0.4])
        self.bn.cpt(self.work_hours).fillWith([0.4, 0.4, 0.2])
        self.bn.cpt(self.stress_level).fillWith([0.3, 0.4, 0.3])
        self.bn.cpt(self.team_size).fillWith([0.3, 0.4, 0.3])
        self.bn.cpt(self.growth_speed).fillWith([0.3, 0.4, 0.3])
        self.bn.cpt(self.expected_salary).fillWith([0.3, 0.4, 0.3])

        # Career conditional probabilities
        # We'll create a more realistic distribution based on common career paths
        career_probs = np.zeros((3, 3, 3, 3, 3, 3, 3, 6))  # Dimensions for each variable state

        # Set base probabilities for each career based on education
        edu_probs = {
            0: [0.25, 0.15, 0.15, 0.20, 0.15, 0.10],  # Bachelor's
            1: [0.20, 0.25, 0.15, 0.15, 0.15, 0.10],  # Master's
            2: [0.15, 0.30, 0.15, 0.10, 0.20, 0.10]  # PhD
        }

        # Fill the CPT with base probabilities and add small variations
        for i in range(3):  # education
            for j in range(3):  # work_location
                for k in range(3):  # work_hours
                    for l in range(3):  # stress_level
                        for m in range(3):  # team_size
                            for n in range(3):  # growth_speed
                                for o in range(3):  # expected_salary
                                    # Start with base probabilities from education
                                    probs = edu_probs[i].copy()

                                    # Add small random variations based on other factors
                                    variations = np.random.uniform(-0.05, 0.05, 6)
                                    probs = np.array(probs) + variations

                                    # Ensure probabilities are valid
                                    probs = np.maximum(probs, 0.01)  # Minimum probability
                                    probs = probs / probs.sum()  # Normalize

                                    career_probs[i, j, k, l, m, n, o, :] = probs

        # Fill the CPT
        self.bn.cpt(self.career).fillWith(career_probs.flatten())

    def get_recommendations(self, preferences: Dict[str, str], top_n: int = 3) -> List[Dict]:
        """
        Generate career recommendations based on preferences

        Args:
            preferences: Dictionary of user preferences
            top_n: Number of top recommendations to return

        Returns:
            List of dictionaries containing career titles and match percentages
        """
        try:
            ie = gum.LazyPropagation(self.bn)

            # Convert preference values to indices
            evidence = {}
            for var_name, value in preferences.items():
                var = getattr(self, var_name)
                var_labels = self.bn.variable(var).labels()
                if value in var_labels:
                    evidence[var] = var_labels.index(value)

            # Set evidence
            ie.setEvidence(evidence)
            ie.makeInference()

            # Get posterior probabilities for careers
            career_probs = ie.posterior(self.career)
            career_labels = self.bn.variable(self.career).labels()

            # Create recommendations list
            recommendations = []
            for i, career in enumerate(career_labels):
                match_percentage = round(float(career_probs[i]) * 100)
                recommendations.append({
                    'title': career,
                    'match': match_percentage
                })

            # Sort by match percentage and return top N
            return sorted(recommendations, key=lambda x: x['match'], reverse=True)[:top_n]

        except Exception as e:
            print(f"Error in Bayesian inference: {e}")
            return []


def main():
    """Test the Bayesian network"""
    network = CareerBayesianNetwork()

    # Test preferences
    preferences = {
        'education': "Master's",
        'work_location': 'Remote',
        'work_hours': 'Flexible Hours',
        'stress_level': 'Moderate',
        'team_size': 'Medium Team',
        'growth_speed': 'Fast Track',
        'expected_salary': 'Mid Range'
    }

    recommendations = network.get_recommendations(preferences)
    print("\nCareer Recommendations:")
    for rec in recommendations:
        print(f"{rec['title']}: {rec['match']}% match")


if __name__ == "__main__":
    main()