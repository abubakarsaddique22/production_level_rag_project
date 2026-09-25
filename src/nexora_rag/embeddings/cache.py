import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self,
        text: str,
        model_name: str,
    ) -> str:
        cache_input = f"{model_name}:{text}"

        return hashlib.sha256(
            cache_input.encode("utf-8")
        ).hexdigest()

    def get(
        self,
        text: str,
        model_name: str,
    ) -> list[float] | None:
        cache_key = self._get_cache_key(
            text,
            model_name,
        )

        cache_path = self.cache_dir / f"{cache_key}.json"

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data["embedding"]

        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def set(
        self,
        text: str,
        model_name: str,
        embedding: list[float],
    ) -> None:
        cache_key = self._get_cache_key(
            text,
            model_name,
        )

        cache_path = self.cache_dir / f"{cache_key}.json"

        data = {
            "model": model_name,
            "text_hash": hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            "embedding": embedding,
        }

        temporary_path = cache_path.with_suffix(".tmp")

        with open(
            temporary_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
            )

        temporary_path.replace(cache_path)

    def delete(
        self,
        text: str,
        model_name: str,
    ) -> None:
        cache_key = self._get_cache_key(
            text,
            model_name,
        )

        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            cache_path.unlink()

    def clear(self) -> None:
        if not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def size(self) -> int:
        if not self.cache_dir.exists():
            return 0

        return len(
            list(self.cache_dir.glob("*.json"))
        )