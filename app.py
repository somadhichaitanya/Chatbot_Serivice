from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from schemas import ChatRequest, ChatResponse, TicketCreate
from db import SessionLocal, init_db, ConversationLog, Ticket
from nlp.advanced_nlp import AdvancedNLP
from nlp.ner import extract_entities
from nlp.faq import FAQRetriever
from nlp.policy import DialogPolicy
import json, os

# Optional OpenAI support (if OPENAI_API_KEY is set in env)
OPENAI_AVAILABLE = False
OPENAI_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API")
if OPENAI_KEY:
    try:
        import openai
        openai.api_key = OPENAI_KEY
        OPENAI_AVAILABLE = True
    except Exception:
        OPENAI_AVAILABLE = False

app = FastAPI(title='Customer Service Chatbot API', version='2.0.0 (advanced)')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Initialize DB & components
init_db()
nlp = AdvancedNLP()
faq = FAQRetriever()
policy = DialogPolicy()

@app.get('/health')
def health():
    return {'status': 'ok', 'mode': 'advanced'}

def get_conversation_history(conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    history = []
    if not conversation_id:
        return history
    try:
        with SessionLocal() as db:
            rows = db.query(ConversationLog).filter(ConversationLog.conversation_id == conversation_id).order_by(ConversationLog.id.desc()).limit(limit).all()
            for r in reversed(rows):
                history.append({'user': r.user_message, 'bot': r.bot_reply})
    except Exception:
        pass
    return history

def openai_reply(message: str, history: List[Dict[str,Any]]) -> str:
    if not OPENAI_AVAILABLE:
        return ""
    try:
        msgs = [{"role": "system", "content": "You are a helpful customer support assistant for an e-commerce store. Be concise and friendly."}]
        for turn in history:
            if 'user' in turn and turn['user']:
                msgs.append({"role": "user", "content": turn['user']})
            if 'bot' in turn and turn['bot']:
                msgs.append({"role": "assistant", "content": turn['bot']})
        msgs.append({"role": "user", "content": message})
        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=msgs, max_tokens=200, temperature=0.2)
        return resp.choices[0].message['content'].strip()
    except Exception:
        return ""

@app.post('/chat', response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message.strip()
    conv_id = req.conversation_id or 'default'

    # 1) classify intent with advanced NLP
    intent, conf, meta = nlp.classify_intent(text)

    # 2) extract regex entities
    entities = extract_entities(text)

    # 3) FAQ fallback
    faq_answer, faq_score = faq.search(text)

    reply = ""
    next_action = None
    ticket_id = None

    # If confidence is low and OpenAI available, ask OpenAI to craft a better response using conversation history
    if conf < 0.50 and OPENAI_AVAILABLE:
        history = get_conversation_history(conv_id, limit=8)
        ai_resp = openai_reply(text, history)
        if ai_resp:
            reply = ai_resp
            # Log and return
            with SessionLocal() as db:
                log = ConversationLog(
                    user_id=req.user_id or 'anonymous',
                    conversation_id=conv_id,
                    user_message=text,
                    bot_reply=reply,
                    intent=intent,
                    confidence=conf,
                    entities=json.dumps(entities)
                )
                db.add(log)
                db.commit()
            return ChatResponse(reply=reply, intent=intent, confidence=conf, entities=entities, faq_answer=faq_answer)

    # Otherwise use existing policy logic (slot filling, FAQ, escalation)
    decision = policy.decide(intent=intent, confidence=conf, entities=entities, faq=(faq_answer, faq_score))
    reply = decision.get('reply', 'Sorry, I could not handle that.')
    next_action = decision.get('next_action')

    # Ticket creation for escalations
    if decision.get('create_ticket'):
        with SessionLocal() as db:
            t = Ticket(
                user_id=req.user_id or 'anonymous',
                conversation_id=conv_id,
                subject=decision.get('ticket_subject', 'Support Request'),
                details=f'User said: {text}\\nEntities: {json.dumps(entities)}'
            )
            db.add(t)
            db.commit()
            db.refresh(t)
            ticket_id = t.id
            reply += f"\\nI created a support ticket for you: #{ticket_id}. Our team will reach out soon."

    # Log conversation
    with SessionLocal() as db:
        log = ConversationLog(
            user_id=req.user_id or 'anonymous',
            conversation_id=conv_id,
            user_message=text,
            bot_reply=reply,
            intent=intent,
            confidence=conf,
            entities=json.dumps(entities)
        )
        db.add(log)
        db.commit()

    return ChatResponse(
        reply=reply,
        intent=intent,
        confidence=conf,
        entities=entities,
        faq_answer=faq_answer,
        next_action=next_action,
        ticket_id=ticket_id
    )
