import json
import os
from typing import List

MCP_FILE = "mcp.json"

class MCPManager:
    def __init__(self):
        self.data = {"stories": []}
        self._load()

    def _load(self):
        if os.path.exists(MCP_FILE):
            with open(MCP_FILE, "r") as f:
                self.data = json.load(f)
        else:
            self._save()

    def _save(self):
        with open(MCP_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_context(self) -> List[str]:
        return self.data.get("stories", [])

    def add_story(self, story: str):
        self.data.setdefault("stories", []).append(story)
        self._save()

    def clear_context(self):
        self.data = {"stories": []}
        self._save()
