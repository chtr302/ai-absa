from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    NEUTRAL = "Neutral"

class Category(str, Enum):
    PERFORMANCE = "PERFORMANCE"
    INTELLIGENCE = "INTELLIGENCE"
    RESOURCES = "RESOURCES"
    BEHAVIOR = "BEHAVIOR"
    TECHNICAL = "TECHNICAL"
    SOFTWARE = "SOFTWARE"
    COMPARATIVE = "COMPARATIVE"

class ABSAQuadruplet(BaseModel):
    """
    Core Data Unit for ASQP (Aspect Sentiment Quadruple Prediction).
    """
    aspect: str = Field(..., description="The target entity being evaluated (e.g., 'Llama 3', 'VRAM').")
    category: Category = Field(..., description="The logical anchor category for the aspect.")
    opinion: str = Field(..., description="The descriptive word expressing the sentiment (e.g., 'blazing fast').")
    sentiment: Sentiment = Field(..., description="Sentiment polarity.")

class ABSAResult(BaseModel):
    """
    Unified result for a comment group.
    Standard interface between KIBAC models and Dashboard.
    """
    id: str = Field(..., description="Unique comment identifier.")
    parent_context: Optional[str] = Field(None, description="Body of the parent comment for implicit aspect resolution.")
    thread_title: str = Field(..., description="Title of the Reddit thread.")
    sentences: List[dict] = Field(..., description="List of sentences with their respective quads.")
    model_name: str = Field("KIBAC-3.0", description="Identifier for the model used.")
    inference_time_ms: Optional[float] = Field(None, description="Inference latency.")
