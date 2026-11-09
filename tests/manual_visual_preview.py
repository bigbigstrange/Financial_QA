"""Isolated browser fixture: temporary SQLite/artifacts; no production models or DB.

Run: .venv/Scripts/python.exe -m tests.manual_visual_preview
"""
import json
from pathlib import Path

from fastapi import Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from types import SimpleNamespace
from tests.integration.test_evidence_api import EvidenceAPITests
from app.core.deps import get_current_user


def main():
    import uvicorn
    fixture = EvidenceAPITests()
    fixture.setUp()
    fixture._add_image()
    app = fixture.client.app

    def user(authorization: str = Header(default='')):
        if authorization != 'Bearer isolated-preview':
            raise HTTPException(401)
        return SimpleNamespace(id=1, role='user')
    app.dependency_overrides[get_current_user] = user

    @app.post('/api/auth/login')
    def login():
        return {'access_token': 'isolated-preview'}

    @app.get('/api/auth/me')
    def me():
        return dict(id=1, username='isolated-preview', email='preview@example.com', role='user')

    @app.get('/api/health')
    def health():
        return dict(ready=True, documents=1, paid_api_disabled=True)

    @app.post('/api/chat')
    def chat():
        return dict(conversation_id=1, assistant_message=dict(id=1,
            content='隔离测试：仅图像证据，未经文字解析／答案验证', evidence=[
                dict(evidence_id='ev-word', text='', source_doc='mixed.docx', page=2)]))

    dist = Path(__file__).resolve().parents[1]/'frontend/dist'
    app.mount('/assets', StaticFiles(directory=dist/'assets'), name='assets')

    @app.get('/{path:path}')
    def frontend(path: str):
        return FileResponse(dist/'index.html')
    try:
        uvicorn.run(app, host='127.0.0.1', port=8017)
    finally:
        fixture.doCleanups()


if __name__ == '__main__':
    main()
