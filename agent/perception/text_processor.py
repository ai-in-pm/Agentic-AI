"""
Text Processor Module
This module handles the perception capabilities of the agent for text input.
"""
import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Processes text input and extracts features, entities, and other relevant information.
    """
    
    def __init__(self):
        """Initialize the text processor with necessary components."""
        logger.info("Initializing TextProcessor")
        # In a real implementation, we would load NLP models here
        self.intent_patterns = {
            "greeting": r"\b(hello|hi|hey|greetings)\b",
            "question": r"\b(what|how|why|when|where|who|can you|could you)\b.*\?",
            "command": r"\b(do|please|can you|could you|would you)\b.*[^?]$",
            "farewell": r"\b(goodbye|bye|see you|farewell)\b"
        }
        
        # Simple sentiment words for demonstration
        self.positive_words = ["good", "great", "excellent", "amazing", "wonderful", "happy", "like", "love"]
        self.negative_words = ["bad", "terrible", "awful", "horrible", "sad", "hate", "dislike", "poor"]
        
    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the input text and extract structured information.
        
        Args:
            text: The input text to process
            context: Optional context information
            
        Returns:
            A dictionary containing the processed information
        """
        logger.info(f"Processing text: {text}")
        
        # Normalize text
        normalized_text = text.lower().strip()
        
        # Tokenize
        tokens = self._tokenize(normalized_text)
        
        # Extract entities
        entities = self._extract_entities(normalized_text)
        
        # Detect intent
        intent, intent_confidence = self._detect_intent(normalized_text)
        
        # Analyze sentiment
        sentiment, sentiment_score = self._analyze_sentiment(normalized_text)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(normalized_text)
        
        # Create feature vector (simplified for demonstration)
        feature_vector = self._create_feature_vector(normalized_text)
        
        # Prepare the result
        result = {
            "original_text": text,
            "normalized_text": normalized_text,
            "tokens": tokens,
            "entities": entities,
            "intent": {
                "type": intent,
                "confidence": intent_confidence
            },
            "sentiment": {
                "label": sentiment,
                "score": sentiment_score
            },
            "key_phrases": key_phrases,
            "feature_vector": feature_vector.tolist(),  # Convert numpy array to list for JSON serialization
            "metadata": {
                "token_count": len(tokens),
                "processing_timestamp": self._get_timestamp()
            }
        }
        
        # Add context if provided
        if context:
            result["context"] = context
        
        logger.info(f"Text processing complete: {intent} intent detected with {intent_confidence:.2f} confidence")
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        # Simple whitespace tokenization for demonstration
        # In a real implementation, we would use a more sophisticated tokenizer
        return text.split()
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        # Simplified entity extraction for demonstration
        # In a real implementation, we would use a named entity recognition model
        entities = []
        
        # Simple pattern matching for dates
        date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        date_matches = re.finditer(date_pattern, text)
        for match in date_matches:
            entities.append({
                "text": match.group(),
                "type": "DATE",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Simple pattern matching for numbers
        number_pattern = r"\b\d+\b"
        number_matches = re.finditer(number_pattern, text)
        for match in number_matches:
            entities.append({
                "text": match.group(),
                "type": "NUMBER",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95
            })
        
        return entities
    
    def _detect_intent(self, text: str) -> tuple:
        """Detect the intent of the text."""
        # Simple rule-based intent detection for demonstration
        # In a real implementation, we would use an intent classification model
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return intent, 0.8
        
        # Default to "statement" with lower confidence
        return "statement", 0.6
    
    def _analyze_sentiment(self, text: str) -> tuple:
        """Analyze the sentiment of the text."""
        # Simple rule-based sentiment analysis for demonstration
        # In a real implementation, we would use a sentiment analysis model
        tokens = self._tokenize(text)
        
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.5 + (positive_count - negative_count) / (len(tokens) * 2)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 0.5 - (negative_count - positive_count) / (len(tokens) * 2)
        else:
            sentiment = "neutral"
            score = 0.5
        
        return sentiment, score
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from the text."""
        # Simplified key phrase extraction for demonstration
        # In a real implementation, we would use a key phrase extraction model
        tokens = self._tokenize(text)
        # Filter out common stop words (simplified)
        stop_words = ["the", "a", "an", "in", "on", "at", "to", "for", "with", "by", "of", "and", "or"]
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 3]
        
        # Return the most frequent tokens as key phrases
        from collections import Counter
        token_counts = Counter(filtered_tokens)
        return [token for token, count in token_counts.most_common(3)]
    
    def _create_feature_vector(self, text: str) -> np.ndarray:
        """Create a feature vector from the text."""
        # Simplified feature extraction for demonstration
        # In a real implementation, we would use word embeddings or other NLP features
        
        # Create a simple bag-of-words vector
        tokens = self._tokenize(text)
        # Use the first 10 tokens or pad with zeros
        vector_size = 10
        vector = np.zeros(vector_size)
        
        for i, token in enumerate(tokens[:vector_size]):
            # Simple hash-based feature
            vector[i] = hash(token) % 100 / 100.0
            
        return vector
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
