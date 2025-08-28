from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_chat_greet():
    r = client.post('/chat', json={'message': 'hello'})
    assert r.status_code == 200
    data = r.json()
    assert 'reply' in data
    assert isinstance(data['intent'], str)
