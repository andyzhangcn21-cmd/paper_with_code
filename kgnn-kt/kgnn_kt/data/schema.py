from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Problem(BaseModel):
    problem_id: str
    text: str
    code: str

class Interaction(BaseModel):
    user_id: str
    problem_id: str
    correct: int = Field(ge=0, le=1)
    timestamp: int

class UserProfile(BaseModel):
    user_id: str
    features: Dict[str, Any] = Field(default_factory=dict)
