from fastapi.testclient import TestClient
from api import main as api_main
from api.main import app
import json

def test_health():
    client=TestClient(app)
    r=client.get('/health')
    assert r.status_code==200
    assert 'status' in r.json()

def test_blank_rejected():
    client=TestClient(app)
    r=client.post('/predict',json={'question':' ','context':'evidence','answer':'answer'})
    assert r.status_code==422

def test_saved_threshold_is_used_when_no_environment_override(tmp_path,monkeypatch):
    threshold_path=tmp_path/'selected_threshold.json'
    threshold_path.write_text(json.dumps({'threshold':0.37}),encoding='utf-8')
    monkeypatch.setattr(api_main,'THRESHOLD_PATH',threshold_path)
    monkeypatch.delenv('RAGGUARD_THRESHOLD',raising=False)
    assert api_main.get_threshold()==0.37

def test_environment_threshold_overrides_saved_threshold(tmp_path,monkeypatch):
    threshold_path=tmp_path/'selected_threshold.json'
    threshold_path.write_text(json.dumps({'threshold':0.37}),encoding='utf-8')
    monkeypatch.setattr(api_main,'THRESHOLD_PATH',threshold_path)
    monkeypatch.setenv('RAGGUARD_THRESHOLD','0.62')
    assert api_main.get_threshold()==0.62
