from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    user_id: Optional[str] = "anonymous"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    entities: Dict[str, Any] = {}
    faq_answer: Optional[str] = None
    next_action: Optional[str] = None
    ticket_id: Optional[int] = None

class TicketCreate(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    subject: str
    details: str
