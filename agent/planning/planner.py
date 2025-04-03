"""
Planner Module
This module handles the planning capabilities of the agent.
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class Planner:
    """
    Generates and evaluates action plans based on reasoning results.
    """
    
    def __init__(self):
        """Initialize the planner with necessary components."""
        logger.info("Initializing Planner")
        
        # Available actions (simplified for demonstration)
        self.available_actions = {
            "retrieve_information": {
                "description": "Retrieve information from knowledge base",
                "parameters": ["query", "sources"],
                "cost": 0.3,
                "time_estimate": 0.5  # in seconds
            },
            "generate_response": {
                "description": "Generate a text response",
                "parameters": ["content", "tone"],
                "cost": 0.2,
                "time_estimate": 0.3
            },
            "ask_clarification": {
                "description": "Ask for clarification from the user",
                "parameters": ["question"],
                "cost": 0.1,
                "time_estimate": 0.2
            },
            "perform_calculation": {
                "description": "Perform a mathematical calculation",
                "parameters": ["expression"],
                "cost": 0.4,
                "time_estimate": 0.6
            },
            "search_external": {
                "description": "Search external sources for information",
                "parameters": ["query", "sources"],
                "cost": 0.7,
                "time_estimate": 1.5
            }
        }
        
        # Goal templates (simplified for demonstration)
        self.goal_templates = {
            "question": {
                "primary": "Provide accurate information",
                "secondary": ["Be concise", "Cite sources"]
            },
            "command": {
                "primary": "Execute requested action correctly",
                "secondary": ["Confirm completion", "Report any issues"]
            },
            "greeting": {
                "primary": "Establish rapport",
                "secondary": ["Be friendly", "Set expectations"]
            },
            "farewell": {
                "primary": "End conversation positively",
                "secondary": ["Summarize if needed", "Offer future assistance"]
            },
            "statement": {
                "primary": "Acknowledge and respond appropriately",
                "secondary": ["Show understanding", "Provide relevant information"]
            }
        }
    
    def create_plan(self, reasoning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an action plan based on reasoning results.
        
        Args:
            reasoning_data: The analysis results from the reasoning module
            
        Returns:
            A dictionary containing the action plan
        """
        logger.info("Creating action plan")
        
        # Extract relevant information from reasoning data
        context = reasoning_data.get("context", {})
        inferences = reasoning_data.get("inferences", [])
        hypotheses = reasoning_data.get("hypotheses", [])
        confidence = reasoning_data.get("confidence", 0.5)
        
        # Define goals based on context
        goals = self._define_goals(context, inferences)
        
        # Generate possible action sequences
        action_sequences = self._generate_action_sequences(goals, context, confidence)
        
        # Evaluate and rank action sequences
        ranked_sequences = self._evaluate_action_sequences(action_sequences, goals, confidence)
        
        # Select the best action sequence
        best_sequence = ranked_sequences[0] if ranked_sequences else {"actions": [], "score": 0}
        
        # Create contingency plans if confidence is low
        contingency_plans = []
        if confidence < 0.7:
            contingency_plans = self._create_contingency_plans(best_sequence, context)
        
        # Prepare the result
        result = {
            "goals": goals,
            "primary_plan": {
                "actions": best_sequence["actions"],
                "estimated_success": best_sequence["score"],
                "estimated_completion_time": sum(action.get("time_estimate", 0) for action in best_sequence["actions"])
            },
            "alternative_plans": [
                {
                    "actions": seq["actions"],
                    "estimated_success": seq["score"],
                    "estimated_completion_time": sum(action.get("time_estimate", 0) for action in seq["actions"])
                }
                for seq in ranked_sequences[1:3]  # Include top 2 alternatives
            ] if len(ranked_sequences) > 1 else [],
            "contingency_plans": contingency_plans,
            "metadata": {
                "planning_confidence": confidence,
                "planning_timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Plan created with {len(best_sequence['actions'])} actions and {len(contingency_plans)} contingencies")
        return result
    
    def _define_goals(self, context: Dict[str, Any], inferences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Define goals based on context and inferences."""
        # Determine the primary interaction type
        interaction_type = context.get("interaction_type", "statement")
        
        # Get goal template
        template = self.goal_templates.get(interaction_type, self.goal_templates["statement"])
        
        # Create goals structure
        goals = {
            "primary": template["primary"],
            "secondary": template["secondary"].copy(),
            "success_criteria": []
        }
        
        # Add success criteria based on interaction type
        if interaction_type == "question":
            goals["success_criteria"].append({
                "criterion": "Information accuracy",
                "threshold": 0.9,
                "weight": 0.6
            })
            goals["success_criteria"].append({
                "criterion": "Information completeness",
                "threshold": 0.8,
                "weight": 0.4
            })
        elif interaction_type == "command":
            goals["success_criteria"].append({
                "criterion": "Action completion",
                "threshold": 0.95,
                "weight": 0.7
            })
            goals["success_criteria"].append({
                "criterion": "User satisfaction",
                "threshold": 0.8,
                "weight": 0.3
            })
        else:
            goals["success_criteria"].append({
                "criterion": "Response appropriateness",
                "threshold": 0.8,
                "weight": 0.5
            })
            goals["success_criteria"].append({
                "criterion": "User engagement",
                "threshold": 0.7,
                "weight": 0.5
            })
        
        # Add additional goals based on inferences
        for inference in inferences:
            if "user is seeking information" in inference.get("conclusion", "").lower():
                if "Information accuracy" not in [c["criterion"] for c in goals["success_criteria"]]:
                    goals["success_criteria"].append({
                        "criterion": "Information accuracy",
                        "threshold": 0.9,
                        "weight": 0.6
                    })
            
            if "user has a negative attitude" in inference.get("conclusion", "").lower():
                goals["secondary"].append("Address user concerns")
                goals["success_criteria"].append({
                    "criterion": "Emotional support",
                    "threshold": 0.8,
                    "weight": 0.4
                })
        
        return goals
    
    def _generate_action_sequences(self, goals: Dict[str, Any], context: Dict[str, Any], confidence: float) -> List[Dict[str, Any]]:
        """Generate possible action sequences to achieve the goals."""
        action_sequences = []
        
        # Determine the interaction type
        interaction_type = context.get("interaction_type", "statement")
        
        # Generate action sequence for questions
        if interaction_type == "question":
            # Standard information retrieval sequence
            standard_sequence = {
                "actions": [
                    {
                        "action_id": "retrieve_information",
                        "parameters": {
                            "query": "based on user input",
                            "sources": ["knowledge_base"]
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["retrieve_information"]["time_estimate"]
                    },
                    {
                        "action_id": "generate_response",
                        "parameters": {
                            "content": "retrieved information",
                            "tone": context.get("emotional_tone", "neutral")
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                    }
                ],
                "description": "Standard information retrieval and response"
            }
            action_sequences.append(standard_sequence)
            
            # Enhanced information retrieval with external search
            if confidence < 0.8:  # If we're not very confident, search external sources
                enhanced_sequence = {
                    "actions": [
                        {
                            "action_id": "retrieve_information",
                            "parameters": {
                                "query": "based on user input",
                                "sources": ["knowledge_base"]
                            },
                            "priority": "high",
                            "time_estimate": self.available_actions["retrieve_information"]["time_estimate"]
                        },
                        {
                            "action_id": "search_external",
                            "parameters": {
                                "query": "based on user input",
                                "sources": ["web", "specialized_databases"]
                            },
                            "priority": "medium",
                            "time_estimate": self.available_actions["search_external"]["time_estimate"]
                        },
                        {
                            "action_id": "generate_response",
                            "parameters": {
                                "content": "combined information",
                                "tone": context.get("emotional_tone", "neutral")
                            },
                            "priority": "high",
                            "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                        }
                    ],
                    "description": "Enhanced information retrieval with external search"
                }
                action_sequences.append(enhanced_sequence)
        
        # Generate action sequence for commands
        elif interaction_type == "command":
            command_sequence = {
                "actions": [
                    {
                        "action_id": "perform_calculation" if "calculation" in " ".join(context.get("topics", [])).lower() else "retrieve_information",
                        "parameters": {
                            "expression" if "calculation" in " ".join(context.get("topics", [])).lower() else "query": "based on user input",
                            "sources": ["knowledge_base"] if "calculation" not in " ".join(context.get("topics", [])).lower() else None
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["perform_calculation" if "calculation" in " ".join(context.get("topics", [])).lower() else "retrieve_information"]["time_estimate"]
                    },
                    {
                        "action_id": "generate_response",
                        "parameters": {
                            "content": "action result",
                            "tone": context.get("emotional_tone", "neutral")
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                    }
                ],
                "description": "Execute command and report result"
            }
            action_sequences.append(command_sequence)
        
        # Generate action sequence for greetings
        elif interaction_type == "greeting":
            greeting_sequence = {
                "actions": [
                    {
                        "action_id": "generate_response",
                        "parameters": {
                            "content": "greeting",
                            "tone": "friendly"
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                    }
                ],
                "description": "Respond to greeting"
            }
            action_sequences.append(greeting_sequence)
        
        # Generate action sequence for farewells
        elif interaction_type == "farewell":
            farewell_sequence = {
                "actions": [
                    {
                        "action_id": "generate_response",
                        "parameters": {
                            "content": "farewell",
                            "tone": "friendly"
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                    }
                ],
                "description": "Respond to farewell"
            }
            action_sequences.append(farewell_sequence)
        
        # Generate action sequence for statements
        else:
            # If confidence is low, ask for clarification
            if confidence < 0.6:
                clarification_sequence = {
                    "actions": [
                        {
                            "action_id": "ask_clarification",
                            "parameters": {
                                "question": "based on ambiguity"
                            },
                            "priority": "high",
                            "time_estimate": self.available_actions["ask_clarification"]["time_estimate"]
                        }
                    ],
                    "description": "Ask for clarification"
                }
                action_sequences.append(clarification_sequence)
            
            # Standard acknowledgment
            acknowledgment_sequence = {
                "actions": [
                    {
                        "action_id": "generate_response",
                        "parameters": {
                            "content": "acknowledgment",
                            "tone": context.get("emotional_tone", "neutral")
                        },
                        "priority": "medium",
                        "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                    }
                ],
                "description": "Acknowledge statement"
            }
            action_sequences.append(acknowledgment_sequence)
        
        return action_sequences
    
    def _evaluate_action_sequences(self, action_sequences: List[Dict[str, Any]], goals: Dict[str, Any], confidence: float) -> List[Dict[str, Any]]:
        """Evaluate and rank action sequences based on goals and confidence."""
        ranked_sequences = []
        
        for sequence in action_sequences:
            # Calculate base score based on alignment with goals
            base_score = self._calculate_goal_alignment(sequence, goals)
            
            # Adjust score based on confidence
            adjusted_score = base_score * (0.5 + 0.5 * confidence)
            
            # Calculate cost
            cost = sum(self.available_actions.get(action["action_id"], {}).get("cost", 0.5) for action in sequence["actions"])
            
            # Calculate time estimate
            time_estimate = sum(action.get("time_estimate", 0.5) for action in sequence["actions"])
            
            # Calculate final score (higher is better)
            # We want high alignment with goals, high confidence, low cost, and reasonable time
            final_score = adjusted_score * (1.0 - 0.3 * cost) * (1.0 - 0.2 * min(time_estimate, 2.0) / 2.0)
            
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, final_score))
            
            # Add some small random variation to break ties
            final_score += np.random.normal(0, 0.02)
            final_score = max(0.0, min(1.0, final_score))
            
            ranked_sequences.append({
                "actions": sequence["actions"],
                "description": sequence.get("description", ""),
                "score": final_score,
                "cost": cost,
                "time_estimate": time_estimate
            })
        
        # Sort by score (descending)
        ranked_sequences.sort(key=lambda x: x["score"], reverse=True)
        
        return ranked_sequences
    
    def _calculate_goal_alignment(self, sequence: Dict[str, Any], goals: Dict[str, Any]) -> float:
        """Calculate how well an action sequence aligns with the goals."""
        # Simplified calculation for demonstration
        # In a real implementation, this would be more sophisticated
        
        # Check if the sequence has actions that address the primary goal
        primary_goal = goals.get("primary", "").lower()
        
        # Initialize alignment score
        alignment_score = 0.5  # Start with a neutral score
        
        # Check action alignment with primary goal
        for action in sequence["actions"]:
            action_id = action["action_id"]
            
            # Retrieve information aligns with information-related goals
            if action_id == "retrieve_information" and "information" in primary_goal:
                alignment_score += 0.3
            
            # Generate response aligns with response-related goals
            elif action_id == "generate_response" and ("respond" in primary_goal or "information" in primary_goal):
                alignment_score += 0.3
            
            # Ask clarification aligns with understanding-related goals
            elif action_id == "ask_clarification" and "understand" in primary_goal:
                alignment_score += 0.3
            
            # Perform calculation aligns with calculation-related goals
            elif action_id == "perform_calculation" and "calculation" in primary_goal:
                alignment_score += 0.3
            
            # Search external aligns with information-related goals
            elif action_id == "search_external" and "information" in primary_goal:
                alignment_score += 0.2
        
        # Check if the sequence addresses success criteria
        for criterion in goals.get("success_criteria", []):
            criterion_name = criterion.get("criterion", "").lower()
            weight = criterion.get("weight", 0.5)
            
            # Information accuracy
            if criterion_name == "information accuracy":
                if any(a["action_id"] == "retrieve_information" for a in sequence["actions"]):
                    alignment_score += 0.2 * weight
                if any(a["action_id"] == "search_external" for a in sequence["actions"]):
                    alignment_score += 0.3 * weight
            
            # Action completion
            elif criterion_name == "action completion":
                if any(a["action_id"] in ["perform_calculation", "retrieve_information"] for a in sequence["actions"]):
                    alignment_score += 0.3 * weight
            
            # User satisfaction or engagement
            elif "user" in criterion_name:
                if any(a["action_id"] == "generate_response" and a["parameters"].get("tone") in ["friendly", "empathetic"] for a in sequence["actions"]):
                    alignment_score += 0.2 * weight
        
        # Ensure alignment score is between 0 and 1
        alignment_score = max(0.0, min(1.0, alignment_score))
        
        return alignment_score
    
    def _create_contingency_plans(self, best_sequence: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create contingency plans for the best action sequence."""
        contingency_plans = []
        
        # Create a contingency plan for low confidence in information retrieval
        if any(action["action_id"] == "retrieve_information" for action in best_sequence["actions"]):
            contingency_plan = {
                "trigger": "Low confidence in retrieved information (below 0.7)",
                "actions": [
                    {
                        "action_id": "search_external",
                        "parameters": {
                            "query": "based on user input",
                            "sources": ["web", "specialized_databases"]
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["search_external"]["time_estimate"]
                    },
                    {
                        "action_id": "generate_response",
                        "parameters": {
                            "content": "external information",
                            "tone": context.get("emotional_tone", "neutral")
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                    }
                ]
            }
            contingency_plans.append(contingency_plan)
        
        # Create a contingency plan for user dissatisfaction
        contingency_plan = {
            "trigger": "User expresses dissatisfaction",
            "actions": [
                {
                    "action_id": "ask_clarification",
                    "parameters": {
                        "question": "about user needs"
                    },
                    "priority": "high",
                    "time_estimate": self.available_actions["ask_clarification"]["time_estimate"]
                }
            ]
        }
        contingency_plans.append(contingency_plan)
        
        # Create a contingency plan for execution errors
        if any(action["action_id"] in ["perform_calculation", "retrieve_information", "search_external"] for action in best_sequence["actions"]):
            contingency_plan = {
                "trigger": "Execution error in primary actions",
                "actions": [
                    {
                        "action_id": "generate_response",
                        "parameters": {
                            "content": "error explanation",
                            "tone": "apologetic"
                        },
                        "priority": "high",
                        "time_estimate": self.available_actions["generate_response"]["time_estimate"]
                    },
                    {
                        "action_id": "ask_clarification",
                        "parameters": {
                            "question": "for alternative approach"
                        },
                        "priority": "medium",
                        "time_estimate": self.available_actions["ask_clarification"]["time_estimate"]
                    }
                ]
            }
            contingency_plans.append(contingency_plan)
        
        return contingency_plans
