import redis

from nexora_rag.core import cache


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.last_ttl = None

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex=None):
        self.store[key] = value
        self.last_ttl = ex


class BrokenRedis:
    def get(self, key):
        raise redis.RedisError("down")

    def set(self, key, value, ex=None):
        raise redis.RedisError("down")


# ---------- make_key ----------

def test_key_ignores_case_spacing_and_department_order():
    a = cache.make_key("  Hotel   CAP? ", ["HR", "Finance"])
    b = cache.make_key("hotel cap?", ["Finance", "HR"])
    assert a == b


def test_key_changes_with_departments():
    assert cache.make_key("q", ["HR"]) != cache.make_key("q", ["Finance"])


# ---------- get / set ----------

def test_set_then_get_roundtrip(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(cache, "_client", fake)

    cache.set_cached_answer("k", "Answer [1].", [{"id": 1}])

    assert cache.get_cached_answer("k") == {"answer": "Answer [1].", "sources": [{"id": 1}]}
    assert fake.last_ttl == cache.settings.answer_cache_ttl_seconds


def test_non_ascii_answer_survives(monkeypatch):
    monkeypatch.setattr(cache, "_client", FakeRedis())
    cache.set_cached_answer("k", "caf\u00e9 [1]", [])
    assert cache.get_cached_answer("k")["answer"] == "caf\u00e9 [1]"


def test_missing_key_returns_none(monkeypatch):
    monkeypatch.setattr(cache, "_client", FakeRedis())
    assert cache.get_cached_answer("nope") is None


# ---------- Redis down: the app must keep working ----------

def test_redis_down_on_get_returns_none(monkeypatch):
    monkeypatch.setattr(cache, "_client", BrokenRedis())
    assert cache.get_cached_answer("k") is None


def test_redis_down_on_set_does_not_raise(monkeypatch):
    monkeypatch.setattr(cache, "_client", BrokenRedis())
    cache.set_cached_answer("k", "a", [])  # must not raise