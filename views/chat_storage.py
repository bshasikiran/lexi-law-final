"""
Chat Storage — Redis (Upstash) with local JSON file fallback.
If Redis is unreachable, chats are stored locally in a JSON file.

Note: Redis is used as a cache/backup for individual chat data.
The chat LIST (names) is always maintained in the local JSON index
because Upstash may restrict the KEYS command.
"""
import json
import os
import threading

# Resolve paths relative to the project root (parent of views/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOTENV_PATH = os.path.join(_PROJECT_ROOT, ".env")
CHATS_FILE = os.path.join(_PROJECT_ROOT, "chat_history.json")
_lock = threading.Lock()

# Try to initialize Redis
_redis = None
_redis_available = False


def _init_redis():
    global _redis, _redis_available
    try:
        from upstash_redis import Redis
        from dotenv import load_dotenv

        # Explicitly load from project root .env with override=True
        load_dotenv(_DOTENV_PATH, override=True)

        url = os.getenv("UPSTASH_REDIS_URL", "")
        token = os.getenv("UPSTASH_REDIS_TOKEN", "")

        if url and token:
            _redis = Redis(url=url, token=token)
            _redis.ping()
            _redis_available = True
            print(f"[ChatStorage] ✅ Redis connected successfully to {url}")
        else:
            print("[ChatStorage] No Redis credentials found, using local storage.")
            print(f"[ChatStorage] Looked for .env at: {_DOTENV_PATH}")
    except Exception as e:
        print(f"[ChatStorage] ⚠️ Redis unavailable ({e}), using local JSON storage.")
        _redis_available = False


# Try Redis once at startup
try:
    _init_redis()
except Exception:
    _redis_available = False


# === Local JSON helpers ===

def _load_all_local() -> dict:
    """Load all chats from local JSON file."""
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_all_local(data: dict) -> None:
    """Save all chats to local JSON file."""
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _parse_redis_value(data) -> dict:
    """Parse a value from Redis into a dict, handling str/dict types."""
    if data is None:
        return None
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


# === Public API ===

def load_chat(chat_name: str) -> dict:
    """Load chat data. Tries Redis first, falls back to local."""
    if _redis_available:
        try:
            data = _redis.get(chat_name)
            parsed = _parse_redis_value(data)
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"[ChatStorage] Redis read error: {e}")

    # Local fallback
    with _lock:
        all_chats = _load_all_local()
        return all_chats.get(chat_name, {"generated": [], "past": []})


def save_chat(chat_name: str, chat_data: dict) -> None:
    """Save chat data. Writes to both local and Redis."""
    # Always save locally (reliable, and maintains the chat index)
    with _lock:
        all_chats = _load_all_local()
        all_chats[chat_name] = chat_data
        _save_all_local(all_chats)

    # Also save to Redis
    if _redis_available:
        try:
            _redis.set(chat_name, json.dumps(chat_data))
        except Exception as e:
            print(f"[ChatStorage] Redis write error: {e}")


def create_new_chat() -> str:
    """Create a new empty chat and return the chat name."""
    with _lock:
        all_chats = _load_all_local()
        existing_count = len([k for k in all_chats.keys() if k.startswith("Chat ")])
        new_name = f"Chat {existing_count + 1}"

        # Avoid collisions
        while new_name in all_chats:
            existing_count += 1
            new_name = f"Chat {existing_count + 1}"

        all_chats[new_name] = {"generated": [], "past": []}
        _save_all_local(all_chats)

    # Also store in Redis
    if _redis_available:
        try:
            _redis.set(new_name, json.dumps({"generated": [], "past": []}))
        except Exception:
            pass

    return new_name


def get_chat_list() -> list:
    """Get sorted list of chat names."""
    with _lock:
        all_chats = _load_all_local()
        names = sorted([k for k in all_chats.keys() if k.startswith("Chat ")])
    return names


def delete_chat(chat_name: str) -> None:
    """Delete a chat from both local and Redis."""
    with _lock:
        all_chats = _load_all_local()
        if chat_name in all_chats:
            del all_chats[chat_name]
            _save_all_local(all_chats)

    if _redis_available:
        try:
            _redis.delete(chat_name)
        except Exception:
            pass
