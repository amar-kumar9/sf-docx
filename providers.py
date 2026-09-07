"""Pluggable LLM providers for the Salesforce docs agent.

The retrieval pipeline does not need a model. Synthesis does. This module
detects whatever the current machine has (local runtime or a cloud key) and
exposes a single get_llm() entry point.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

ROOT_DIR = Path(__file__).resolve().parent
ENV_PATH = ROOT_DIR / ".env"

AUTO_CLOUD_ORDER = ("groq", "openai", "anthropic", "google")
AUTO_LOCAL_ORDER = ("ollama", "lmstudio", "openai_compatible")
PROVIDER_CHOICES = ("auto",) + AUTO_CLOUD_ORDER + AUTO_LOCAL_ORDER

DEFAULT_MODELS = {
    "groq": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "google": os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
    "ollama": os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
    "lmstudio": os.getenv("LM_STUDIO_MODEL", "local-model"),
    "openai_compatible": os.getenv("LLM_MODEL", "local-model"),
}


class LLMResponse:
    def __init__(self, content: str):
        self.content = content


def _role(message: Any) -> str:
    kind = str(getattr(message, "type", "") or message.__class__.__name__).lower()
    if "system" in kind:
        return "system"
    if "ai" in kind or "assistant" in kind:
        return "assistant"
    return "user"


class ChatCompletionsLLM:
    """OpenAI-compatible /v1/chat/completions client (Groq, OpenAI, LM Studio, Ollama)."""

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.endpoint = f"{self.base_url}/chat/completions"
        else:
            self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout

    def invoke(self, messages):
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": _role(item), "content": getattr(item, "content", str(item))}
                for item in messages
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return LLMResponse(content)


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def invoke(self, messages):
        system = ""
        body = []
        for item in messages:
            role = _role(item)
            text = getattr(item, "content", str(item))
            if role == "system":
                system = text
            else:
                body.append({"role": "user" if role == "user" else "assistant", "content": text})
        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": body or [{"role": "user", "content": ""}],
        }
        if system:
            payload["system"] = system
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        blocks = response.json().get("content") or []
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        return LLMResponse(text)


def _ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://localhost:11434/").rstrip("/") + "/"


def _lm_studio_url() -> str:
    return os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/")


def _openai_compatible_url() -> str:
    return (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").rstrip("/")


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False


def _key(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _model_for(provider: str, tier: str) -> str:
    override = (os.getenv("LLM_MODEL") or "").strip()
    if override:
        return override
    if provider == "ollama" and tier == "fast":
        return os.getenv("OLLAMA_MODEL_FAST", "gemma3:1b")
    return DEFAULT_MODELS.get(provider, "local-model")


def detect_providers() -> list[dict[str, Any]]:
    rows = [
        {
            "id": "groq",
            "label": "Groq",
            "kind": "cloud",
            "available": bool(_key("GROQ_API_KEY")),
            "detail": "Set GROQ_API_KEY",
        },
        {
            "id": "openai",
            "label": "OpenAI",
            "kind": "cloud",
            "available": bool(_key("OPENAI_API_KEY")),
            "detail": "Set OPENAI_API_KEY",
        },
        {
            "id": "anthropic",
            "label": "Anthropic",
            "kind": "cloud",
            "available": bool(_key("ANTHROPIC_API_KEY")),
            "detail": "Set ANTHROPIC_API_KEY",
        },
        {
            "id": "google",
            "label": "Google Gemini",
            "kind": "cloud",
            "available": bool(_key("GOOGLE_API_KEY", "GEMINI_API_KEY")),
            "detail": "Set GOOGLE_API_KEY or GEMINI_API_KEY",
        },
        {
            "id": "ollama",
            "label": "Ollama (local)",
            "kind": "local",
            "available": _http_ok(_ollama_url()),
            "detail": f"{_ollama_url()} — install from https://ollama.com",
        },
        {
            "id": "lmstudio",
            "label": "LM Studio (local)",
            "kind": "local",
            "available": _http_ok(_lm_studio_url() + "/models"),
            "detail": f"{_lm_studio_url()} — enable the local server in LM Studio",
        },
        {
            "id": "openai_compatible",
            "label": "Custom OpenAI-compatible",
            "kind": "local",
            "available": bool(_openai_compatible_url()),
            "detail": "Set LLM_BASE_URL plus LLM_API_KEY if the server requires it",
        },
    ]
    return rows


def _auto_order() -> list[str]:
    prefer_local = os.getenv("LLM_PREFER_LOCAL", "").lower() in {"1", "true", "yes", "on"}
    if prefer_local:
        return list(AUTO_LOCAL_ORDER + AUTO_CLOUD_ORDER)
    return list(AUTO_CLOUD_ORDER + AUTO_LOCAL_ORDER)


def selected_provider_id() -> str:
    raw = (os.getenv("LLM_PROVIDER") or "auto").strip().lower()
    if raw not in PROVIDER_CHOICES:
        return "auto"
    return raw


def _make_provider(provider: str, tier: str):
    model = _model_for(provider, tier)
    if provider == "groq":
        key = _key("GROQ_API_KEY")
        if not key:
            return None
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(model=model, groq_api_key=key, temperature=0, max_tokens=2048)
        except Exception:
            return ChatCompletionsLLM("https://api.groq.com/openai/v1", key, model)
    if provider == "openai":
        key = _key("OPENAI_API_KEY", "LLM_API_KEY")
        if not key:
            return None
        return ChatCompletionsLLM("https://api.openai.com/v1", key, model)
    if provider == "anthropic":
        key = _key("ANTHROPIC_API_KEY")
        if not key:
            return None
        return AnthropicLLM(key, model)
    if provider == "google":
        key = _key("GOOGLE_API_KEY", "GEMINI_API_KEY")
        if not key:
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
        except Exception:
            return None
    if provider == "ollama":
        if not _http_ok(_ollama_url()):
            return None
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=0, base_url=_ollama_url().rstrip("/"))
        except Exception:
            return ChatCompletionsLLM(_ollama_url().rstrip("/") + "/v1", "ollama", model)
    if provider == "lmstudio":
        if not _http_ok(_lm_studio_url() + "/models"):
            return None
        return ChatCompletionsLLM(_lm_studio_url(), _key("LM_STUDIO_API_KEY") or "lm-studio", model)
    if provider == "openai_compatible":
        base = _openai_compatible_url()
        if not base:
            return None
        return ChatCompletionsLLM(base, _key("LLM_API_KEY", "OPENAI_API_KEY"), model)
    return None


def get_llm(tier: str = "standard") -> tuple[str | None, Any]:
    preferred = selected_provider_id()
    order = [preferred] if preferred != "auto" else _auto_order()
    detected = {row["id"]: row for row in detect_providers()}
    for provider in order:
        row = detected.get(provider)
        if preferred == "auto" and row and not row["available"]:
            continue
        try:
            llm = _make_provider(provider, tier)
        except Exception:
            llm = None
        if llm is not None:
            model = _model_for(provider, tier)
            label = f"{provider}/{model}"
            return label, llm
    return None, None


def provider_status() -> dict[str, Any]:
    rows = detect_providers()
    label, _ = get_llm("fast")
    return {
        "selected": selected_provider_id(),
        "active": label,
        "prefer_local": os.getenv("LLM_PREFER_LOCAL", "").lower() in {"1", "true", "yes", "on"},
        "providers": rows,
        "env_file": str(ENV_PATH),
        "env_file_exists": ENV_PATH.exists(),
    }


def _parse_env_lines(text: str) -> list[str]:
    return text.splitlines()


def save_provider_settings(updates: dict[str, str]) -> Path:
    """Merge key=value pairs into the local .env without deleting other keys."""
    cleaned = {str(key): str(value).strip() for key, value in updates.items() if str(key).strip()}
    existing_lines: list[str] = []
    if ENV_PATH.exists():
        existing_lines = _parse_env_lines(ENV_PATH.read_text(encoding="utf-8"))
    keys_written = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in cleaned:
            new_lines.append(f"{key}={cleaned[key]}")
            keys_written.add(key)
        else:
            new_lines.append(line)
    for key, value in cleaned.items():
        if key not in keys_written:
            new_lines.append(f"{key}={value}")
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENV_PATH.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    for key, value in cleaned.items():
        if value:
            os.environ[key] = value
        elif key in os.environ:
            os.environ.pop(key, None)
    return ENV_PATH


def apply_runtime_provider(provider: str, api_key: str = "", base_url: str = "", model: str = "") -> dict[str, str]:
    provider = (provider or "auto").strip().lower()
    if provider not in PROVIDER_CHOICES:
        provider = "auto"
    updates = {"LLM_PROVIDER": provider}
    if model.strip():
        updates["LLM_MODEL"] = model.strip()
    if base_url.strip():
        if provider == "ollama":
            updates["OLLAMA_URL"] = base_url.strip()
        elif provider == "lmstudio":
            updates["LM_STUDIO_URL"] = base_url.strip()
        else:
            updates["LLM_BASE_URL"] = base_url.strip()
    key = api_key.strip()
    if key:
        key_env = {
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai_compatible": "LLM_API_KEY",
            "lmstudio": "LM_STUDIO_API_KEY",
        }.get(provider)
        if key_env:
            updates[key_env] = key
    save_provider_settings(updates)
    return updates


def host_hint(url: str | None) -> str:
    if not url:
        return ""
    host = urlparse(url).hostname or url
    return host
