import re

ORDER_ID_RE = re.compile(r'\b\d{3}-\d{7,8}-\d{7,8}\b')  # e.g., 123-4567890-1234567
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}')
PHONE_RE = re.compile(r'\b\+?\d[\d\s\-]{7,}\b')

def extract_entities(text: str):
    entities = {}
    order_ids = ORDER_ID_RE.findall(text)
    if order_ids:
        entities['order_id'] = order_ids[0]
    email = EMAIL_RE.findall(text)
    if email:
        entities['email'] = email[0]
    phone = PHONE_RE.findall(text)
    if phone:
        entities['phone'] = phone[0]
    return entities
