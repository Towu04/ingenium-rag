import json
from pathlib import Path


class StateTracker:
    def __init__(self, path: str | Path = "data/state.json") -> None:
        self.path = Path(path)
        self._state = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"State tracker must contain an object: {self.path}")
        return {str(key): str(digest) for key, digest in value.items()}

    def has_changed(self, document_id: str, content_hash: str) -> bool:
        return self._state.get(document_id) != content_hash

    def record(self, document_id: str, content_hash: str) -> None:
        self._state[document_id] = content_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self._state, file, indent=2, sort_keys=True)
