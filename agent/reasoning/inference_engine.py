"""
Inference Engine Module
This module handles the reasoning capabilities of the agent.
"""
import logging
import os
from typing import Dict, Any, List, Optional
import numpy as np
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Analyzes processed data using logical inference and domain knowledge.
    """

    def __init__(self):
        """Initialize the inference engine with necessary components."""
        logger.info("Initializing InferenceEngine")
        # In a real implementation, we would load reasoning models here

        # Simple rule base for demonstration
        self.rules = [
            {
                "condition": lambda data: data["intent"]["type"] == "greeting",
                "conclusion": "The user is starting a conversation",
                "confidence": 0.95
            },
            {
                "condition": lambda data: data["intent"]["type"] == "question",
                "conclusion": "The user is seeking information",
                "confidence": 0.9
            },
            {
                "condition": lambda data: data["intent"]["type"] == "command",
                "conclusion": "The user wants the agent to perform an action",
                "confidence": 0.85
            },
            {
                "condition": lambda data: data["intent"]["type"] == "farewell",
                "conclusion": "The user is ending the conversation",
                "confidence": 0.95
            },
            {
                "condition": lambda data: data["sentiment"]["label"] == "positive",
                "conclusion": "The user has a positive attitude",
                "confidence": 0.8
            },
            {
                "condition": lambda data: data["sentiment"]["label"] == "negative",
                "conclusion": "The user has a negative attitude",
                "confidence": 0.8
            }
        ]

        # Domain knowledge base (simplified for demonstration)
        self.knowledge_base = {
            "greeting_responses": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?"
            ],
            "farewell_responses": [
                "Goodbye! Have a great day!",
                "Farewell! It was nice chatting with you.",
                "See you later! Feel free to return if you need more help."
            ],
            "fallback_responses": [
                "I'm not sure I understand. Could you please rephrase?",
                "I'm still learning. Can you provide more details?",
                "I don't have enough information to respond properly."
            ]
        }

    def analyze(self, perception_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the processed data using logical inference.

        Args:
            perception_data: The processed data from the perception module

        Returns:
            A dictionary containing the analysis results
        """
        logger.info("Analyzing perception data")

        # Apply rules to draw inferences
        inferences = self._apply_rules(perception_data)

        # Determine the context
        context = self._determine_context(perception_data, inferences)

        # Generate hypotheses
        hypotheses = self._generate_hypotheses(perception_data, inferences)

        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(inferences)

        # Use OpenAI for enhanced reasoning if API key is available
        ai_inferences = []
        if os.getenv("OPENAI_API_KEY"):
            try:
                ai_inferences = self._get_ai_inferences(perception_data)
                # Add AI-generated inferences to our list
                inferences.extend(ai_inferences)
                # Recalculate uncertainty with AI input
                uncertainty = self._calculate_uncertainty(inferences)
            except Exception as e:
                logger.warning(f"Error using OpenAI API: {e}")

        # Prepare the result
        result = {
            "inferences": inferences,
            "context": context,
            "hypotheses": hypotheses,
            "uncertainty": uncertainty,
            "confidence": 1.0 - uncertainty,
            "ai_enhanced": len(ai_inferences) > 0,
            "metadata": {
                "rule_count": len(self.rules),
                "ai_inference_count": len(ai_inferences),
                "processing_timestamp": self._get_timestamp()
            }
        }

        logger.info(f"Analysis complete: {len(inferences)} inferences drawn with {result['confidence']:.2f} confidence")
        return result

    def _apply_rules(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply reasoning rules to the data."""
        inferences = []

        for rule in self.rules:
            try:
                if rule["condition"](data):
                    inferences.append({
                        "conclusion": rule["conclusion"],
                        "confidence": rule["confidence"],
                        "source": "rule-based-inference"
                    })
            except Exception as e:
                logger.warning(f"Error applying rule: {e}")

        # Add additional inferences based on entities
        if "entities" in data and data["entities"]:
            inferences.append({
                "conclusion": "The input contains specific entities that may be important",
                "confidence": 0.7,
                "source": "entity-analysis"
            })

        # Add inferences based on key phrases
        if "key_phrases" in data and data["key_phrases"]:
            key_phrases_str = ", ".join(data["key_phrases"])
            inferences.append({
                "conclusion": f"The key topics are related to: {key_phrases_str}",
                "confidence": 0.75,
                "source": "key-phrase-analysis"
            })

        return inferences

    def _determine_context(self, data: Dict[str, Any], inferences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the context of the interaction."""
        context = {
            "interaction_type": data["intent"]["type"],
            "emotional_tone": data["sentiment"]["label"],
            "topics": data.get("key_phrases", []),
            "relevant_entities": [entity["text"] for entity in data.get("entities", [])]
        }

        # Determine if this is a follow-up or new conversation
        if "context" in data and data["context"]:
            context["conversation_stage"] = "follow-up"
            context["previous_context"] = data["context"]
        else:
            context["conversation_stage"] = "new"

        return context

    def _generate_hypotheses(self, data: Dict[str, Any], inferences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate hypotheses based on the data and inferences."""
        hypotheses = []

        # Generate hypothesis based on intent
        intent_type = data["intent"]["type"]
        intent_confidence = data["intent"]["confidence"]

        if intent_type == "question":
            hypotheses.append({
                "hypothesis": "User needs information",
                "probability": intent_confidence * 0.9,
                "evidence": ["intent classification", "question structure"]
            })
        elif intent_type == "command":
            hypotheses.append({
                "hypothesis": "User wants an action performed",
                "probability": intent_confidence * 0.85,
                "evidence": ["intent classification", "command structure"]
            })
        elif intent_type == "greeting":
            hypotheses.append({
                "hypothesis": "User is initiating conversation",
                "probability": intent_confidence * 0.95,
                "evidence": ["intent classification", "greeting pattern"]
            })
        elif intent_type == "farewell":
            hypotheses.append({
                "hypothesis": "User is ending conversation",
                "probability": intent_confidence * 0.95,
                "evidence": ["intent classification", "farewell pattern"]
            })

        # Generate hypothesis based on sentiment
        sentiment = data["sentiment"]["label"]
        sentiment_score = data["sentiment"]["score"]

        if sentiment == "positive" and sentiment_score > 0.7:
            hypotheses.append({
                "hypothesis": "User is satisfied or happy",
                "probability": sentiment_score * 0.8,
                "evidence": ["sentiment analysis", "positive language"]
            })
        elif sentiment == "negative" and sentiment_score < 0.3:
            hypotheses.append({
                "hypothesis": "User is dissatisfied or frustrated",
                "probability": (1 - sentiment_score) * 0.8,
                "evidence": ["sentiment analysis", "negative language"]
            })

        return hypotheses

    def _calculate_uncertainty(self, inferences: List[Dict[str, Any]]) -> float:
        """Calculate the uncertainty of the analysis."""
        if not inferences:
            return 0.5  # Maximum uncertainty when no inferences

        # Average the confidence of all inferences and subtract from 1 to get uncertainty
        avg_confidence = sum(inf["confidence"] for inf in inferences) / len(inferences)

        # Add some random noise to simulate real-world uncertainty
        noise = np.random.normal(0, 0.05)
        uncertainty = 1.0 - avg_confidence + noise

        # Ensure uncertainty is between 0 and 1
        uncertainty = max(0.0, min(1.0, uncertainty))

        return uncertainty

    def _get_ai_inferences(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get inferences from OpenAI API."""
        logger.info("Getting AI inferences from OpenAI")

        # Extract relevant information from perception data
        text = data.get("original_text", "")
        intent = data.get("intent", {}).get("type", "unknown")
        sentiment = data.get("sentiment", {}).get("label", "neutral")
        entities = [entity.get("text", "") for entity in data.get("entities", [])]
        key_phrases = data.get("key_phrases", [])

        # Prepare the prompt for OpenAI
        prompt = f"""Analyze the following user input and provide logical inferences:

User input: "{text}"

Detected intent: {intent}
Detected sentiment: {sentiment}
Detected entities: {', '.join(entities) if entities else 'None'}
Key phrases: {', '.join(key_phrases) if key_phrases else 'None'}

Provide 3-5 logical inferences about the user's needs, intentions, or state of mind based on this input. Format each inference as a JSON object with 'conclusion' and 'confidence' (0.0-1.0) fields."""

        try:
            # Call OpenAI API
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI reasoning assistant that analyzes user inputs and provides logical inferences. Respond only with the requested JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Parse the inferences from the response
            import json
            import re

            # Try to extract JSON objects from the text
            inferences = []
            try:
                # First try to parse the entire response as JSON
                json_objects = json.loads(response_text)
                if isinstance(json_objects, list):
                    inferences = json_objects
                else:
                    inferences = [json_objects]
            except json.JSONDecodeError:
                # If that fails, try to extract JSON objects using regex
                pattern = r'\{[^\{\}]*"conclusion"[^\{\}]*"confidence"[^\{\}]*\}|\{[^\{\}]*"confidence"[^\{\}]*"conclusion"[^\{\}]*\}'
                matches = re.findall(pattern, response_text)
                for match in matches:
                    try:
                        inference = json.loads(match)
                        inferences.append(inference)
                    except json.JSONDecodeError:
                        continue

            # Format the inferences properly
            formatted_inferences = []
            for inference in inferences:
                if "conclusion" in inference and "confidence" in inference:
                    formatted_inferences.append({
                        "conclusion": inference["conclusion"],
                        "confidence": float(inference["confidence"]),
                        "source": "ai-inference"
                    })

            logger.info(f"Generated {len(formatted_inferences)} AI inferences")
            return formatted_inferences

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return []

    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
