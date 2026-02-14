#!/usr/bin/env python3
"""
Unified AI client for StreamSniped.

Behavior:
- Try OpenRouter free models in order (configurable), minimal retries per model.
- If all OpenRouter options fail, fallback to Gemini 2.0 Flash (paid model).
- Vision: Gemini 2.0 Flash first, then OpenRouter fallback.
- Quiet retry logging by default (toggle via QUIET_RETRY_LOGS=false).

Usage:
    from src.ai_client import call_llm, call_llm_vision
    text = call_llm(prompt, max_tokens=500, temperature=0.3, request_tag="chapter_analysis")
    vision_result = call_llm_vision(prompt, images, request_tag="vision_analysis")
"""

from __future__ import annotations

import os
import time
import random
from typing import List, Optional, Tuple, Dict
import json
import requests
from dotenv import load_dotenv
from threading import BoundedSemaphore, Lock
from contextlib import contextmanager

# Load environment variables from config/streamsniped.env
load_dotenv("config/streamsniped.env")


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


# ------------------------- Global Concurrency/Throttle -------------------------

_LLM_MAX_PARALLEL = int(os.getenv("LLM_MAX_PARALLEL", "8"))
_LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S", "0.0"))
_LLM_SEM = BoundedSemaphore(max(1, _LLM_MAX_PARALLEL))
_LLM_LAST_CALL_TS = 0.0
_LLM_TIME_LOCK = Lock()



def _maybe_throttle():
    """Enforce a minimal interval between outbound LLM requests across threads.

    Disabled when LLM_MIN_INTERVAL_S == 0.0.
    """
    global _LLM_LAST_CALL_TS
    if _LLM_MIN_INTERVAL_S <= 0.0:
        return
    with _LLM_TIME_LOCK:
        now = time.monotonic()
        wait = _LLM_MIN_INTERVAL_S - (now - _LLM_LAST_CALL_TS)
        if wait > 0.0:
            time.sleep(wait)
        _LLM_LAST_CALL_TS = time.monotonic()


@contextmanager
def _llm_slot():
    """Context manager that limits global parallel LLM calls and applies throttle."""
    _LLM_SEM.acquire()
    try:
        _maybe_throttle()
        yield
    finally:
        _LLM_SEM.release()


def _get_free_models_from_env() -> List[str]:
    models_env = os.getenv("OPENROUTER_FREE_MODELS", "").strip()
    if models_env:
        return [m.strip() for m in models_env.split(",") if m.strip()]
    # Default rotation order (free tiers) - reduced list for faster cycling
    return [
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-r1-0528:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "openai/gpt-oss-20b:free",
        "z-ai/glm-4.5-air:free",
    ]


def _quiet() -> bool:
    return os.getenv("QUIET_RETRY_LOGS", "false").lower() in ["true", "1", "yes"]


def _post_openrouter(prompt: str, model: str, api_key: str, max_tokens: int, temperature: float, timeout: int) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo/streamsniped",
        "X-Title": "StreamSniped Unified AI Client",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        if not _quiet():
            print(f" OpenRouter error ({model}): {resp.status_code}")
        return None
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def _call_openrouter_round_robin(prompt: str, max_tokens: int, temperature: float, timeout: int) -> Optional[str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        if not _quiet():
            print("❌ OPENROUTER_API_KEY not found in environment")
        return None
    
    if not _quiet():
        print(f"🔑 Using OpenRouter API key: {api_key[:10]}...")
    models = _get_free_models_from_env()
    per_model_retries = int(os.getenv("OPENROUTER_PER_MODEL_RETRIES", "1"))
    base_delay = float(os.getenv("OPENROUTER_BASE_DELAY", "1.0"))

    if not _quiet():
        print(f" OpenRouter configured; trying {len(models)} free models")

    for model in models:
        for attempt in range(1, per_model_retries + 1):
            # if not _quiet():
            #     print(f"🤖 {model} (attempt {attempt}/{per_model_retries})")
            # elif attempt == 1:
            #     print(f"🤖 Trying {model.split('/')[-1]}...")

            try:
                text = _post_openrouter(prompt, model, api_key, max_tokens, temperature, timeout)
                if text:
                    # if not _quiet():
                    #     print(f" {model} OK")
                    # else:
                    #     print(f" OpenRouter OK: {model.split('/')[-1]}")
                    return text
            except Exception as e:
                if not _quiet():
                    print(f" {model} failed: {e}")

            if attempt < per_model_retries:
                delay = base_delay * attempt + random.uniform(0, 0.3)
                if not _quiet():
                    print(f" Waiting {delay:.1f}s before retry...")
                time.sleep(delay)

    return None


def _call_openai_direct(prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
    """Call OpenAI API directly as a fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if not _quiet():
            print("❌ OPENAI_API_KEY not found in environment")
        return None
    
    if not _quiet():
        print(f"🔑 Using OpenAI API key: {api_key[:10]}...")
    
    try:
        if not _quiet():
            print("🔄 Using OpenAI API...")
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if not _quiet():
            print(" OpenAI OK")
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        if not _quiet():
            print(f" OpenAI Direct failed: {e}")
        return None

def _call_ollama_json(prompt: str,
                      schema: Dict,
                      model: str = None,
                      num_ctx: int = None,
                      num_predict: int = 256,
                      temperature: float = 0.25,
                      top_p: float = 0.9,
                      top_k: int = 64,
                      repeat_penalty: float = 1.15,
                      timeout: int = 60) -> Optional[Dict]:
    """
    Call Ollama with structured JSON output, robust to server failures.

    Strategy:
    1) Skip /api/chat entirely (complex schemas crash llama runner with exit status 2).
    2) Use /api/generate with format:"json" and embed the schema in the prompt.
    3) Never raise; return None on total failure so callers can degrade gracefully.
    """
    gen_base = os.getenv("OLLAMA_URL", "http://localhost:11434")
    gen_url = f"{gen_base.rstrip('/')}/api/generate"
    model = model or os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
    num_ctx = num_ctx or int(os.getenv("LOCAL_NUM_CTX", "16384"))

    # Use generate endpoint with format:"json" and schema embedded in prompt
    try:
        # Create a simple, clear prompt that asks for JSON in the expected format
        constrained_prompt = (
            f"{prompt}\n\n"
            "Return JSON ONLY with this exact format: "
            '{"labels": ["label1", "label2", "label3"]}'
        )
        
        gen_body = {
            "model": model,
            "prompt": constrained_prompt,
            "stream": False,
            "format": "json",
            # Remove options that crash llama runner - use minimal options only
            "options": {
                "temperature": temperature,
            },
        }
        
        # Try with format:json first
        resp = requests.post(gen_url, json=gen_body, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = (data.get("response") or "").strip()
            if text:
                try:
                    result = json.loads(text)
                    if isinstance(result, dict) and "labels" in result and isinstance(result["labels"], list):
                        return result
                except Exception:
                    pass
        
        # Fallback without format enforcement
        if not _quiet():
            print(f" Ollama generate with format returned {resp.status_code}; trying without format")
        
        loose_body = dict(gen_body)
        loose_body.pop("format", None)
        resp2 = requests.post(gen_url, json=loose_body, timeout=timeout)
        if resp2.status_code != 200:
            if not _quiet():
                print(f" Ollama generate returned {resp2.status_code}")
            return None
            
        data2 = resp2.json()
        text2 = (data2.get("response") or "").strip()
        if not text2:
            return None
            
        # Try to extract first JSON object from the text
        s = text2
        if s.startswith("```"):
            try:
                s = s.split("```", 2)[1]
            except Exception:
                pass
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(s[start:end+1])
                if isinstance(result, dict) and "labels" in result and isinstance(result["labels"], list):
                    return result
            except Exception:
                pass
        return None
        
    except Exception as e:
        if not _quiet():
            print(f" Ollama generate error: {e}")
        return None

def _call_ollama(prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
    """Call Ollama API directly."""
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    
    if not _quiet():
        print(f"🦙 Using Ollama: {model} at {ollama_url}")
    
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            }
        }
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json=payload,
            timeout=120  # Ollama can be slower
        )
        
        if response.status_code != 200:
            if not _quiet():
                print(f" Ollama error: {response.status_code} - {response.text}")
            return None
            
        data = response.json()
        result = data.get("response", "").strip()
        
        if result:
            if not _quiet():
                print(" Ollama OK")
            return result
        else:
            if not _quiet():
                print(" Ollama returned empty response")
            return None
            
    except Exception as e:
        if not _quiet():
            print(f" Ollama failed: {e}")
        return None


def _call_gemini_direct(prompt: str, max_tokens: int, temperature: float) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        if not _quiet():
            print("❌ GEMINI_API_KEY not found in environment")
        return None
    
    if not _quiet():
        print(f"🔑 Using Gemini API key: {api_key[:10]}...")
    try:
        if not _quiet():
            print("🔄 Falling back to Gemini Direct API...")
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_tokens),
            )
        )
        if not _quiet():
            print(" Gemini OK")
        return (response.text or "").strip()
    except Exception as e:
        if not _quiet():
            print(f" Gemini Direct failed: {e}")
        return None


def call_gemini_3_flash(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.4,
    request_tag: Optional[str] = None,
) -> str:
    """Call Gemini 3 Flash Preview directly for high-quality title/content generation.
    
    This is the primary model for all title, timestamp, and creative content generation.
    Uses "gemini-3-flash-preview" for best balance of speed and quality.
    
    Returns the generated text or raises RuntimeError if failed.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")
    
    if request_tag and not _quiet():
        print(f"📨 Gemini 3 Flash request: {request_tag}")
    
    with _llm_slot():
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Use Gemini 3 Flash Preview
            model_name = os.getenv("GEMINI_TITLE_MODEL", "gemini-3-flash-preview")
            model = genai.GenerativeModel(model_name)
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=float(temperature),
                    max_output_tokens=int(max_tokens),
                )
            )
            
            text = (response.text or "").strip()
            if text:
                if not _quiet():
                    print(f"✅ Gemini 3 Flash OK ({model_name})")
                return text
            
            raise RuntimeError("Gemini 3 Flash returned empty response")
            
        except Exception as e:
            if not _quiet():
                print(f"❌ Gemini 3 Flash failed: {e}")
            raise RuntimeError(f"Gemini 3 Flash failed: {e}")


def call_llm(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    timeout: int = 60,
    request_tag: Optional[str] = None,
) -> str:
    """Call LLM with OpenAI primary, Gemini fallback.

    Returns the generated text or raises RuntimeError if all options fail.
    """
    if request_tag and not _quiet():
        print(f"📨 LLM request: {request_tag}")

    with _llm_slot():
        # Try OpenAI first (fast and reliable)
        text = _call_openai_direct(prompt, max_tokens, temperature)
        if text:
            return text

        # Fallback to Gemini 2.0 Flash
        if not _quiet():
            print("🔄 Falling back to Gemini 2.0 Flash...")
        text = _call_gemini_direct(prompt, max_tokens, temperature)
        if text:
            return text

    print(" All AI options failed after trying OpenAI and Gemini 2.0 Flash")
    raise RuntimeError("All AI options failed")




def _call_openai_vision(
    prompt: str,
    images: List[Tuple[str, bytes, str]],
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> Optional[str]:
    """Call OpenAI vision (gpt-4o-mini) directly."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import base64
        import openai
        client = openai.OpenAI(api_key=api_key)
        content: list = [{"type": "text", "text": prompt}]
        for _, img_bytes, mime_type in images:
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            data_url = f"data:{mime_type};base64,{b64}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def _call_gemini_vision(
    prompt: str,
    images: List[Tuple[str, bytes, str]],
    max_tokens: int = 700,
    temperature: float = 0.2,
    request_tag: Optional[str] = None,
) -> Optional[str]:
    """Call Gemini vision API directly.

    Robustly extracts text from candidates/parts; returns None if no text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    try:
        if not _quiet():
            print("🔄 Using Gemini 2.5 Flash for vision...")
        
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Convert images to PIL format for Gemini
        from PIL import Image
        import io
        
        content_parts = [prompt]
        for name, img_bytes, mime_type in images:
            img = Image.open(io.BytesIO(img_bytes))
            content_parts.append(img)
        
        response = model.generate_content(
            content_parts,
            generation_config=genai.types.GenerationConfig(
                temperature=float(temperature),
                max_output_tokens=int(max_tokens),
            )
        )
        # Prefer response.text if available and non-empty
        try:
            txt = (getattr(response, "text", "") or "").strip()
        except Exception:
            txt = ""
        if not txt:
            # Parse candidates -> content.parts for text
            candidates = getattr(response, "candidates", []) or []
            parts_text: list[str] = []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", []) if content is not None else []
                for part in parts:
                    try:
                        part_text = str(getattr(part, "text", "") or "").strip()
                    except Exception:
                        part_text = ""
                    if part_text:
                        parts_text.append(part_text)
            txt = " ".join(parts_text).strip()
        if txt:
            if not _quiet():
                print(" Gemini Vision OK")
            return txt
        return None
        
    except Exception as e:
        if not _quiet():
            print(f" Gemini Vision failed: {e}")
        return None


def call_llm_ollama(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    request_tag: Optional[str] = None,
) -> str:
    """Call Ollama LLM directly with fallback to other models.

    Returns the generated text or raises RuntimeError if all options fail.
    """
    if request_tag and not _quiet():
        print(f"📨 Ollama request: {request_tag}")

    with _llm_slot():
        # Try Ollama first
        text = _call_ollama(prompt, max_tokens, temperature)
        if text:
            return text

        # Fallback to other models
        if not _quiet():
            print("🔄 Ollama failed, falling back to other models...")
        
        # Try OpenAI
        text = _call_openai_direct(prompt, max_tokens, temperature)
        if text:
            return text

        # Try Gemini
        text = _call_gemini_direct(prompt, max_tokens, temperature)
        if text:
            return text

    print(" All AI options failed after trying Ollama, OpenAI, and Gemini")
    raise RuntimeError("All AI options failed")


def call_llm_vision(
    prompt: str,
    images: List[Tuple[str, bytes, str]],
    # images: list of (name, image_bytes, mime_type) e.g. ("frame", b"...", "image/png")
    max_tokens: int = 700,
    temperature: float = 0.2,
    request_tag: Optional[str] = None,
) -> str:
    """Call vision models with OpenAI (gpt-4o-mini) first, then Gemini 2.5 Flash fallback.

    Returns empty string on total failure (no exception).
    """
    if request_tag and not _quiet():
        print(f"📨 Vision request: {request_tag}")
    
    with _llm_slot():
        # Try OpenAI first
        if not _quiet():
            print("🔄 Using OpenAI gpt-4o-mini vision...")
        result = _call_openai_vision(prompt, images, max_tokens=min(max_tokens, 200), temperature=temperature)
        if result:
            if not _quiet():
                print("✅ OpenAI vision succeeded")
            return result

        # Fallback to Gemini 2.5 Flash
        if not _quiet():
            print("🔄 Falling back to Gemini 2.5 Flash vision...")
        result = _call_gemini_vision(prompt, images, max_tokens, temperature, request_tag)
        if result:
            if not _quiet():
                print("✅ Gemini 2.5 Flash vision succeeded")
            return result

    if not _quiet():
        print(" All vision models failed (OpenAI + Gemini)")
    return ""
