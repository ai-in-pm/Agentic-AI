"""
Memory Manager for Exact Recall
This module handles the storage and retrieval of exact memories.
"""
import logging
import json
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the storage and retrieval of exact memories for the agent.
    """

    def __init__(self, memory_file: str = "memories.json"):
        """Initialize the memory manager with necessary components."""
        logger.info("Initializing MemoryManager")
        self.memory_file = memory_file
        self.memories = self._load_memories()

        # Create embeddings for existing memories
        self.embeddings = {}
        for memory_id, memory in self.memories.items():
            self.embeddings[memory_id] = self._create_embedding(memory["content"])

    def store(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new memory.

        Args:
            content: The content to remember
            metadata: Optional metadata about the memory

        Returns:
            The ID of the stored memory
        """
        logger.info(f"Storing new memory: {content[:50]}...")

        # Generate a unique ID for the memory
        memory_id = f"mem_{int(time.time())}_{len(self.memories)}"

        # Create the memory object
        memory = {
            "id": memory_id,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 0,
            "metadata": metadata or {}
        }

        # Store the memory
        self.memories[memory_id] = memory

        # Create and store the embedding
        self.embeddings[memory_id] = self._create_embedding(content)

        # Save the memories to disk
        self._save_memories()

        return memory_id

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a query.

        Args:
            query: The query to search for
            limit: Maximum number of memories to return

        Returns:
            A list of matching memories
        """
        logger.info(f"Retrieving memories for query: {query}")

        if not self.memories:
            logger.info("No memories available")
            return []

        # Create an embedding for the query
        query_embedding = self._create_embedding(query)

        # Calculate similarity scores
        similarities = {}
        for memory_id, embedding in self.embeddings.items():
            similarity = self._calculate_similarity(query_embedding, embedding)
            similarities[memory_id] = similarity

        # Sort memories by similarity
        sorted_memories = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Get the top matching memories
        results = []
        for memory_id, similarity in sorted_memories[:limit]:
            memory = self.memories[memory_id].copy()
            memory["similarity"] = similarity

            # Update access information
            self.memories[memory_id]["access_count"] += 1
            self.memories[memory_id]["last_accessed"] = datetime.now().isoformat()

            results.append(memory)

        # Save the updated memories
        self._save_memories()

        logger.info(f"Retrieved {len(results)} memories")
        return results

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all stored memories.

        Returns:
            A list of all memories
        """
        return list(self.memories.values())

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            True if the memory was deleted, False otherwise
        """
        if memory_id in self.memories:
            del self.memories[memory_id]
            if memory_id in self.embeddings:
                del self.embeddings[memory_id]
            self._save_memories()
            return True
        return False

    def clear(self) -> None:
        """Clear all memories."""
        self.memories = {}
        self.embeddings = {}
        self._save_memories()

    def _load_memories(self) -> Dict[str, Dict[str, Any]]:
        """Load memories from disk."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            return {}

    def _save_memories(self) -> None:
        """Save memories to disk."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving memories: {e}")

    def _create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for a text.

        In a real implementation, this would use a proper embedding model.
        For this demo, we'll use a simple hash-based approach.
        """
        # Simple hash-based embedding (for demonstration only)
        # In a real implementation, use a proper embedding model like sentence-transformers
        embedding_size = 128
        embedding = np.zeros(embedding_size)

        # Create a deterministic but distributed embedding based on the text
        for i, char in enumerate(text):
            embedding[i % embedding_size] += ord(char) / 1000.0

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()

    def _calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate the cosine similarity between two embeddings.

        Args:
            embedding1: The first embedding
            embedding2: The second embedding

        Returns:
            The cosine similarity (between -1 and 1, higher is more similar)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        return float(similarity)

    def _get_timestamp(self) -> str:
        """Get the current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()
