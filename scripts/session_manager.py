"""
Session Manager for Triton Operator Generation System

Manages session lifecycle, including creation, cleanup, and state tracking.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class SessionManager:
    def __init__(self, base_path: str = ".triton-gen/sessions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def create_session(self, task_description: str) -> str:
        session_id = self._generate_session_id()
        session_dir = self.base_path / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        task_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "task_description": task_description,
            "status": "created"
        }
        
        self._write_json(session_dir / "task.json", task_data)
        
        for subdir in ["design", "code", "verification", "optimization", "knowledge"]:
            (session_dir / subdir).mkdir(exist_ok=True)
        
        return session_id
    
    def _generate_session_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"
    
    def get_session_dir(self, session_id: str) -> Path:
        return self.base_path / session_id
    
    def update_task_status(self, session_id: str, status: str, **extra_fields):
        task_file = self.get_session_dir(session_id) / "task.json"
        task_data = self._read_json(task_file)
        task_data["status"] = status
        task_data["updated_at"] = datetime.now().isoformat()
        task_data.update(extra_fields)
        self._write_json(task_file, task_data)
    
    def write_input(self, session_id: str, stage: str, input_data: Dict[str, Any]):
        input_file = self.get_session_dir(session_id) / stage / "input.json"
        self._write_json(input_file, input_data)
    
    def read_output(self, session_id: str, stage: str, filename: str = "output.json") -> Optional[Dict]:
        output_file = self.get_session_dir(session_id) / stage / filename
        if output_file.exists():
            return self._read_json(output_file)
        return None
    
    def cleanup_session(self, session_id: str):
        session_dir = self.get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
    
    def list_sessions(self) -> list:
        sessions = []
        for session_dir in self.base_path.iterdir():
            if session_dir.is_dir():
                task_file = session_dir / "task.json"
                if task_file.exists():
                    sessions.append(self._read_json(task_file))
        return sorted(sessions, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def _write_json(self, filepath: Path, data: Dict):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _read_json(self, filepath: Path) -> Dict:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    sm = SessionManager()
    session_id = sm.create_session("Test session")
    print(f"Created session: {session_id}")
    print(f"Session dir: {sm.get_session_dir(session_id)}")
