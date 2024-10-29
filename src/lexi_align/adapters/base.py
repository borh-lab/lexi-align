from abc import ABC, abstractmethod
from lexi_align.models import TextAlignment

class LLMAdapter(ABC):
    """Base class for LLM adapters."""
    
    @abstractmethod
    def __call__(self, messages: list[dict]) -> TextAlignment:
        """Run the model on the given messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            TextAlignment object with the model's response
        """
        pass
