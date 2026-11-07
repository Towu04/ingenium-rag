import hashlib
import json
from pathlib import Path


class StateLedger:
    def __init__(self, path: str | Path = "data/state.json") -> None:
        self.path = Path(path)
        self._state: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"State ledger must contain an object: {self.path}")
        return {str(key): str(value) for key, value in value.items()}

    @staticmethod
    def digest(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def changed(self, document_id: str, content: bytes) -> bool:
        digest = self.digest(content)
        return self._state.get(document_id) != digest

    def record(self, document_id: str, content: bytes) -> None:
        self._state[document_id] = self.digest(content)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self._state, file, indent=2, sort_keys=True)
