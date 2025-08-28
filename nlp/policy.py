
from typing import Dict, Any, Tuple

SAFE_REPLY = "Sorry — I didn't fully catch that. Could you rephrase or give a bit more detail? 😊"

INTENT_TEMPLATES = {
    'greet': "Hey there! 👋 I'm here to help. You can ask about orders, returns, refunds, shipping, or say 'talk to human' to get an agent.",
    'goodbye': "Thanks for dropping by — have a lovely day! ✨",
    'thanks': "You're welcome! Happy to help.",
    'return_policy': "You can return most items within 30 days of delivery. Want me to start a return? Please share your order ID.",
    'refund_status': "I can check that — please share your order ID so I can fetch the refund status.",
    'product_info': "Tell me the product name or SKU and I'll pull up the details.",
    'shipping_info': "Standard shipping is usually 3–7 business days. Want the exact ETA for your order? Share the order ID.",
    'cancel_order': "I can try to cancel that — share your order ID and I'll check if it's still cancellable.",
    'payment_issue': "Sorry about that. Share the transaction/order ID and I'll look into the payment status.",
    'exchange_request': "Sure — share your order ID and the size/color you'd like to exchange to, and I'll guide you."
}

REQUIRED_SLOTS = {
    'track_order': ['order_id'],
    'refund_status': ['order_id'],
    'cancel_order': ['order_id'],
    'exchange_request': ['order_id'],
}

def _slot_prompt(missing_slots):
    prompts = {
        'order_id': "Please share your order ID (e.g., 123-4567890-1234567).",
        'email': "Can you provide the email used on the order?",
        'phone': "Please share the phone number linked to the order."
    }
    return "\\n".join(prompts[s] for s in missing_slots if s in prompts)

class DialogPolicy:
    def __init__(self, min_conf: float = 0.40, auto_escalate_conf: float = 0.30):
        # min_conf: below this, bot asks clarification or uses FAQ if available
        # auto_escalate_conf: below this, and no strong FAQ, escalate to human
        self.min_conf = min_conf
        self.auto_escalate_conf = auto_escalate_conf

    def decide(self, intent: str, confidence: float, entities: Dict[str, Any], faq: Tuple[str, float]):
        faq_answer, faq_score = faq
        result = {'reply': '', 'next_action': None}

        # If user explicitly asks to escalate, create a ticket.
        if intent == 'escalate':
            result['reply'] = "Got it — connecting you to a human agent. I've raised a ticket and our team will reach out shortly."
            result['create_ticket'] = True
            result['ticket_subject'] = "User requested human agent"
            return result

        # Low confidence: try FAQ, else ask for clarification or escalate if very low.
        if confidence < self.min_conf:
            if faq_answer and faq_score >= 0.35:
                result['reply'] = faq_answer
                return result
            if confidence < self.auto_escalate_conf:
                result['reply'] = "I'm having trouble here — I'll connect you with a human agent so we can resolve this quickly."
                result['create_ticket'] = True
                result['ticket_subject'] = "Auto-escalation - low confidence"
                return result
            result['reply'] = SAFE_REPLY
            return result

        # Normal flow using templates
        if intent in INTENT_TEMPLATES:
            result['reply'] = INTENT_TEMPLATES[intent]

        # Slot filling
        if intent in REQUIRED_SLOTS:
            required = REQUIRED_SLOTS[intent]
            missing = [s for s in required if s not in entities]
            if missing:
                result['reply'] = (result['reply'] + "\\n" if result['reply'] else "") + _slot_prompt(missing)
                result['next_action'] = 'request_slots'
                return result
            # Fulfill simulated actions
            if intent == 'track_order':
                result['reply'] = f"Thanks — tracking order {entities.get('order_id')}... Current status: Out for delivery. ETA: 2 days."
                result['next_action'] = 'fulfill_track_order'
                return result
            if intent == 'refund_status':
                result['reply'] = f"Checking refund for order {entities.get('order_id')}... Status: Processed — funds will reflect in 2–3 business days."
                result['next_action'] = 'fulfill_refund_status'
                return result
            if intent == 'cancel_order':
                result['reply'] = f"Okay — checking if order {entities.get('order_id')} can be cancelled... It looks cancellable. I've started cancellation."
                result['next_action'] = 'fulfill_cancel_order'
                return result

        # If FAQ strong, return it
        if faq_answer and faq_score >= 0.5:
            result['reply'] = faq_answer
            return result

        # Fallback
        if not result['reply']:
            result['reply'] = SAFE_REPLY
        return result
