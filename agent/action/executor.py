"""
Action Executor Module
This module handles the action execution capabilities of the agent.
"""
import logging
import time
from typing import Dict, Any, List, Optional
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class ActionExecutor:
    """
    Executes planned actions and monitors their outcomes.
    """
    
    def __init__(self):
        """Initialize the action executor with necessary components."""
        logger.info("Initializing ActionExecutor")
        
        # Simple knowledge base for demonstration
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
            "information_responses": [
                "Based on my knowledge, {topic} refers to {definition}.",
                "Here's what I know about {topic}: {definition}",
                "Regarding {topic}, I can tell you that {definition}."
            ],
            "clarification_questions": [
                "Could you please provide more details about your request?",
                "I'm not sure I understand completely. Can you elaborate?",
                "To better assist you, could you clarify what you mean by that?"
            ],
            "error_messages": [
                "I apologize, but I encountered an error while processing your request.",
                "Sorry, something went wrong while executing that action.",
                "I'm having trouble completing that task. Could we try a different approach?"
            ],
            "topics": {
                "artificial intelligence": "the simulation of human intelligence in machines that are programmed to think and learn like humans",
                "machine learning": "a subset of AI that enables systems to learn and improve from experience without being explicitly programmed",
                "deep learning": "a subset of machine learning that uses neural networks with many layers to analyze various factors of data",
                "natural language processing": "a field of AI that gives machines the ability to read, understand, and derive meaning from human languages",
                "computer vision": "a field of AI that enables machines to interpret and make decisions based on visual data",
                "robotics": "a field that combines AI, engineering, and computer science to create machines that can perform tasks autonomously",
                "agentic ai": "AI systems that can act autonomously on behalf of users, making decisions and taking actions to achieve specified goals"
            }
        }
        
        # Action handlers
        self.action_handlers = {
            "retrieve_information": self._handle_retrieve_information,
            "generate_response": self._handle_generate_response,
            "ask_clarification": self._handle_ask_clarification,
            "perform_calculation": self._handle_perform_calculation,
            "search_external": self._handle_search_external
        }
        
        # Metrics tracking
        self.metrics = {
            "actions_executed": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "average_execution_time": 0.0
        }
    
    def execute(self, planning_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the planned actions.
        
        Args:
            planning_data: The action plan from the planning module
            
        Returns:
            A dictionary containing the execution results
        """
        logger.info("Executing planned actions")
        
        # Extract the primary plan
        primary_plan = planning_data.get("primary_plan", {})
        actions = primary_plan.get("actions", [])
        
        # Initialize results
        executed_actions = []
        overall_success = True
        final_response = ""
        execution_start_time = time.time()
        
        # Execute each action in sequence
        for action in actions:
            action_id = action.get("action_id")
            parameters = action.get("parameters", {})
            
            logger.info(f"Executing action: {action_id}")
            
            # Execute the action
            action_start_time = time.time()
            try:
                # Get the appropriate handler for this action
                handler = self.action_handlers.get(action_id)
                
                if handler:
                    # Execute the action and get the result
                    action_result = handler(parameters)
                    success = action_result.get("success", False)
                    
                    # If this is a response generation action, save the response
                    if action_id == "generate_response":
                        final_response = action_result.get("response", "")
                    
                    # Update metrics
                    self.metrics["actions_executed"] += 1
                    if success:
                        self.metrics["successful_actions"] += 1
                    else:
                        self.metrics["failed_actions"] += 1
                        overall_success = False
                        
                        # Check if we need to use a contingency plan
                        contingency_plan = self._find_applicable_contingency(planning_data, action_id, success)
                        if contingency_plan:
                            logger.info(f"Applying contingency plan: {contingency_plan.get('trigger')}")
                            contingency_actions = contingency_plan.get("actions", [])
                            
                            # Execute contingency actions
                            for cont_action in contingency_actions:
                                cont_action_id = cont_action.get("action_id")
                                cont_parameters = cont_action.get("parameters", {})
                                
                                cont_handler = self.action_handlers.get(cont_action_id)
                                if cont_handler:
                                    cont_action_result = cont_handler(cont_parameters)
                                    
                                    # If this is a response generation action, save the response
                                    if cont_action_id == "generate_response":
                                        final_response = cont_action_result.get("response", "")
                                    
                                    # Add to executed actions
                                    executed_actions.append({
                                        "action": cont_action_id,
                                        "parameters": cont_parameters,
                                        "status": "completed" if cont_action_result.get("success", False) else "failed",
                                        "result": cont_action_result.get("result", {}),
                                        "time_ms": int((time.time() - action_start_time) * 1000),
                                        "is_contingency": True
                                    })
                else:
                    logger.warning(f"No handler found for action: {action_id}")
                    success = False
                    overall_success = False
                
                # Record the executed action
                executed_actions.append({
                    "action": action_id,
                    "parameters": parameters,
                    "status": "completed" if success else "failed",
                    "result": action_result.get("result", {}),
                    "time_ms": int((time.time() - action_start_time) * 1000),
                    "is_contingency": False
                })
                
            except Exception as e:
                logger.error(f"Error executing action {action_id}: {e}")
                
                # Record the failed action
                executed_actions.append({
                    "action": action_id,
                    "parameters": parameters,
                    "status": "error",
                    "error_message": str(e),
                    "time_ms": int((time.time() - action_start_time) * 1000),
                    "is_contingency": False
                })
                
                overall_success = False
                
                # Check if we need to use a contingency plan for errors
                contingency_plan = self._find_applicable_contingency(planning_data, "error", False)
                if contingency_plan:
                    logger.info(f"Applying error contingency plan: {contingency_plan.get('trigger')}")
                    contingency_actions = contingency_plan.get("actions", [])
                    
                    # Execute contingency actions
                    for cont_action in contingency_actions:
                        cont_action_id = cont_action.get("action_id")
                        cont_parameters = cont_action.get("parameters", {})
                        
                        cont_handler = self.action_handlers.get(cont_action_id)
                        if cont_handler:
                            cont_action_result = cont_handler(cont_parameters)
                            
                            # If this is a response generation action, save the response
                            if cont_action_id == "generate_response":
                                final_response = cont_action_result.get("response", "")
                            
                            # Add to executed actions
                            executed_actions.append({
                                "action": cont_action_id,
                                "parameters": cont_parameters,
                                "status": "completed" if cont_action_result.get("success", False) else "failed",
                                "result": cont_action_result.get("result", {}),
                                "time_ms": int((time.time() - action_start_time) * 1000),
                                "is_contingency": True
                            })
        
        # Calculate total execution time
        total_execution_time = time.time() - execution_start_time
        
        # Update average execution time metric
        if self.metrics["actions_executed"] > 0:
            self.metrics["average_execution_time"] = (
                (self.metrics["average_execution_time"] * (self.metrics["actions_executed"] - len(actions)) + total_execution_time)
                / self.metrics["actions_executed"]
            )
        
        # If no final response was generated, create a default one
        if not final_response:
            if overall_success:
                final_response = "I've processed your request successfully."
            else:
                final_response = "I encountered some issues while processing your request."
        
        # Prepare the result
        result = {
            "executed_actions": executed_actions,
            "success": overall_success,
            "response": final_response,
            "execution_time": total_execution_time,
            "metrics": {
                "total_actions": len(executed_actions),
                "successful_actions": sum(1 for action in executed_actions if action["status"] == "completed"),
                "failed_actions": sum(1 for action in executed_actions if action["status"] != "completed"),
                "contingency_actions": sum(1 for action in executed_actions if action.get("is_contingency", False))
            },
            "metadata": {
                "execution_timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"Execution complete: {result['metrics']['successful_actions']} successful actions, {result['metrics']['failed_actions']} failed actions")
        return result
    
    def _find_applicable_contingency(self, planning_data: Dict[str, Any], action_id: str, success: bool) -> Optional[Dict[str, Any]]:
        """Find an applicable contingency plan based on the action and its success."""
        contingency_plans = planning_data.get("contingency_plans", [])
        
        for plan in contingency_plans:
            trigger = plan.get("trigger", "").lower()
            
            # Check if this contingency plan applies
            if not success and "error" in trigger and action_id in trigger:
                return plan
            elif not success and "execution error" in trigger:
                return plan
            elif not success and "low confidence" in trigger and action_id == "retrieve_information":
                return plan
        
        return None
    
    def _handle_retrieve_information(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the retrieve_information action."""
        query = parameters.get("query", "").lower()
        sources = parameters.get("sources", ["knowledge_base"])
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        # Extract topic from query (simplified)
        topic = None
        for potential_topic in self.knowledge_base["topics"].keys():
            if potential_topic in query:
                topic = potential_topic
                break
        
        # If no specific topic found, use a generic one
        if not topic:
            topic = random.choice(list(self.knowledge_base["topics"].keys()))
        
        # Get the definition
        definition = self.knowledge_base["topics"].get(topic, "I don't have specific information about that.")
        
        # Simulate success with high probability
        success = random.random() < 0.9
        
        return {
            "success": success,
            "result": {
                "topic": topic,
                "definition": definition,
                "sources": sources,
                "confidence": random.uniform(0.7, 0.95) if success else random.uniform(0.3, 0.6)
            }
        }
    
    def _handle_generate_response(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the generate_response action."""
        content_type = parameters.get("content", "").lower()
        tone = parameters.get("tone", "neutral").lower()
        
        # Simulate processing time
        time.sleep(random.uniform(0.05, 0.2))
        
        response = ""
        
        # Generate response based on content type
        if content_type == "greeting":
            response = random.choice(self.knowledge_base["greeting_responses"])
        elif content_type == "farewell":
            response = random.choice(self.knowledge_base["farewell_responses"])
        elif content_type in ["retrieved information", "external information", "combined information"]:
            # Get a random topic and definition for demonstration
            topic = random.choice(list(self.knowledge_base["topics"].keys()))
            definition = self.knowledge_base["topics"][topic]
            
            # Use a template
            template = random.choice(self.knowledge_base["information_responses"])
            response = template.format(topic=topic, definition=definition)
        elif content_type == "error explanation":
            response = random.choice(self.knowledge_base["error_messages"])
        elif content_type == "acknowledgment":
            response = "I understand. " + (
                "That's interesting information." if tone == "neutral" else
                "That's wonderful to hear!" if tone == "positive" else
                "I'm sorry to hear that." if tone == "negative" else
                "Thank you for sharing that with me."
            )
        elif content_type == "action result":
            response = "I've completed the requested action successfully."
        else:
            response = "I've processed your request and here's my response."
        
        # Adjust tone if needed
        if tone == "friendly" and "!" not in response:
            response = response.replace(".", "!")
        elif tone == "apologetic" and "sorry" not in response.lower():
            response = "I'm sorry. " + response
        elif tone == "empathetic":
            response = "I understand how you feel. " + response
        
        # Simulate success with very high probability
        success = random.random() < 0.95
        
        return {
            "success": success,
            "result": {
                "content_type": content_type,
                "tone": tone,
                "length": len(response)
            },
            "response": response
        }
    
    def _handle_ask_clarification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the ask_clarification action."""
        question_type = parameters.get("question", "").lower()
        
        # Simulate processing time
        time.sleep(random.uniform(0.05, 0.15))
        
        # Get a clarification question
        clarification = random.choice(self.knowledge_base["clarification_questions"])
        
        # Adjust based on question type
        if "ambiguity" in question_type:
            clarification = "I'm not sure I understand completely. Could you clarify what you mean?"
        elif "user needs" in question_type:
            clarification = "To better assist you, could you tell me more specifically what you're looking for?"
        elif "alternative approach" in question_type:
            clarification = "I'm having trouble with that approach. Is there another way you'd like me to help you?"
        
        # Simulate success with high probability
        success = random.random() < 0.9
        
        return {
            "success": success,
            "result": {
                "question_type": question_type,
                "clarification": clarification
            },
            "response": clarification
        }
    
    def _handle_perform_calculation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the perform_calculation action."""
        expression = parameters.get("expression", "")
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.4))
        
        # Simulate a calculation result
        try:
            # For demonstration, we'll just generate a random result
            # In a real implementation, we would evaluate the expression
            result = random.uniform(1, 100)
            success = True
        except Exception as e:
            result = None
            success = False
        
        return {
            "success": success,
            "result": {
                "expression": expression,
                "result": result,
                "precision": "high"
            },
            "response": f"The result of the calculation is {result:.2f}." if success else "I couldn't perform that calculation."
        }
    
    def _handle_search_external(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the search_external action."""
        query = parameters.get("query", "").lower()
        sources = parameters.get("sources", ["web"])
        
        # Simulate processing time
        time.sleep(random.uniform(0.3, 0.7))
        
        # Simulate external search results
        # In a real implementation, this would call external APIs
        
        # Extract topic from query (simplified)
        topic = None
        for potential_topic in self.knowledge_base["topics"].keys():
            if potential_topic in query:
                topic = potential_topic
                break
        
        # If no specific topic found, use a generic one
        if not topic:
            topic = random.choice(list(self.knowledge_base["topics"].keys()))
        
        # Get the definition and enhance it for "external" search
        base_definition = self.knowledge_base["topics"].get(topic, "")
        enhanced_definition = base_definition + " This field has seen significant advancements in recent years, with applications across various industries."
        
        # Simulate success with moderate probability (external searches are less reliable)
        success = random.random() < 0.8
        
        return {
            "success": success,
            "result": {
                "topic": topic,
                "information": enhanced_definition,
                "sources": sources,
                "confidence": random.uniform(0.6, 0.9) if success else random.uniform(0.2, 0.5)
            },
            "response": f"According to external sources, {topic} refers to {enhanced_definition}" if success else "I couldn't find reliable information from external sources."
        }
