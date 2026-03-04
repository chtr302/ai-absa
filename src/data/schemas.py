from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class ABSATriplet(BaseModel):
    """
    Core Data Unit for ASTE (Aspect Sentiment Triplet Extraction).
    Supports both Tier 1 (Aspect-Sentiment) and Tier 2 (Aspect-Opinion-Sentiment).
    """
    aspect: str = Field(..., description="The target entity being evaluated (e.g., 'ChatGPT', 'UI').")
    opinion: Optional[str] = Field(None, description="The descriptive word expressing the sentiment (e.g., 'fast', 'hallucinating').")
    sentiment: Sentiment = Field(..., description="Sentiment polarity.")

class ABSADocument(BaseModel):
    """
    Output for a single analyzed text. 
    Standard interface between Models and Dashboard.
    """
    raw_text: str
    triplets: List[ABSATriplet]
    model_name: str = Field("ai-absa-v1", description="Identifier for the model used.")
    inference_time_ms: Optional[float] = Field(None, description="Latency on current hardware (e.g., Mac M1 Pro).")

class BatchABSA(BaseModel):
    """
    Schema for mass inference results, suitable for LMSYS dataset processing.
    """
    results: List[ABSADocument]
    total_count: int
    execution_date: str
