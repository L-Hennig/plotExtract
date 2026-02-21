import base64
import copy
import json
import os
import re
import sys
import traceback
import importlib
import time
import random
import cv2
import numpy as np
from dotenv import load_dotenv
import tempfile
from typing import Optional

try:
    from mistralai import Mistral  # type: ignore
except Exception:
    Mistral = None  # type: ignore

# -----------------------------------------------------------------------------
# Paths and imports that require sys.path adjustments
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PROMPTS_DIR = os.path.join(BASE_DIR, 'prompts')

# Ensure imports work when running as a script (python plot_extract_v2/runner.py ...)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if PROMPTS_DIR not in sys.path:
    sys.path.insert(0, PROMPTS_DIR)

# Import complete schema + constraints
from plot_extract_v2.prompts.complete_extraction_schema import (
    ACCUMULATED_FACTS_SCHEMA,
    SCHEMA_CONSTRAINTS,
)

from plot_extract_v2.stage6a_csv_diagnostics import compute_csv_diagnostics_from_text

# Import extraction tracker
from extraction_tracker import ExtractionTracker

# Load env for API key
load_dotenv(override=True)
API_KEY_MISTRAL_1 = os.getenv("API_KEY_1")
API_KEY_MISTRAL_3 = os.getenv("API_KEY_3")
API_KEY_MISTRAL_4 = os.getenv("API_KEY_4")
API_KEY_GOOGLE = os.getenv("API_KEY_2")

LLM_PROVIDER = (os.getenv("PLOTEXTRACT_LLM_PROVIDER") or "mistral").strip().lower()
LLM_MODEL = (os.getenv("PLOTEXTRACT_LLM_MODEL") or "").strip()
LLM_KEY = (os.getenv("PLOTEXTRACT_LLM_KEY") or "").strip()


def _env_flag(name: str) -> bool:
    v = str(os.getenv(name, "")).strip().lower()
    return v in {"1", "true", "yes", "on"}


RATE_LIMIT_BACKOFF_MODE = _env_flag("PLOTEXTRACT_RATE_LIMIT_BACKOFF_MODE")

try:
    LLM_MIN_INTERVAL_S = float(
        os.getenv("PLOTEXTRACT_LLM_MIN_INTERVAL_S")
        or ("5.0" if RATE_LIMIT_BACKOFF_MODE else "0")
    )
except Exception:
    LLM_MIN_INTERVAL_S = 5.0 if RATE_LIMIT_BACKOFF_MODE else 0.0
LLM_MIN_INTERVAL_S = max(0.0, LLM_MIN_INTERVAL_S)

try:
    BACKOFF_STAGE_SLEEP_S = float(
        os.getenv("PLOTEXTRACT_BACKOFF_STAGE_SLEEP_S")
        or ("0.6" if RATE_LIMIT_BACKOFF_MODE else "0")
    )
except Exception:
    BACKOFF_STAGE_SLEEP_S = 0.6 if RATE_LIMIT_BACKOFF_MODE else 0.0
BACKOFF_STAGE_SLEEP_S = max(0.0, BACKOFF_STAGE_SLEEP_S)

_LAST_LLM_CALL_TS = 0.0


def _maybe_throttle_llm_call():
    """Best-effort pacing to reduce bursty request rate.

    This is intentionally simple: enforce a minimum interval between LLM calls
    within a single runner process.
    """
    global _LAST_LLM_CALL_TS
    if not (LLM_MIN_INTERVAL_S and LLM_MIN_INTERVAL_S > 0):
        return
    now = time.time()
    dt = now - _LAST_LLM_CALL_TS
    if dt < LLM_MIN_INTERVAL_S:
        time.sleep(LLM_MIN_INTERVAL_S - dt)
    _LAST_LLM_CALL_TS = time.time()


def _format_llm_exception_details(err: Exception) -> str:
    """Best-effort extraction of HTTP/status/body details from SDK exceptions.

    Providers/SDKs vary wildly. This tries common patterns without assuming types.
    """
    details: dict = {
        "type": type(err).__name__,
        "module": type(err).__module__,
        "repr": repr(err),
        "str": str(err),
    }

    for attr in (
        "status_code",
        "http_status",
        "code",
        "error_code",
        "message",
        "detail",
        "details",
        "body",
        "text",
    ):
        try:
            if hasattr(err, attr):
                details[attr] = getattr(err, attr)
        except Exception:
            pass

    # Common: err.response (requests/httpx-like)
    try:
        resp = getattr(err, "response", None)
    except Exception:
        resp = None
    if resp is not None:
        try:
            details["response_status_code"] = getattr(resp, "status_code", None)
        except Exception:
            pass
        try:
            hdrs = getattr(resp, "headers", None)
            if hdrs is not None:
                details["response_headers"] = dict(hdrs)
        except Exception:
            pass
        try:
            details["response_text"] = getattr(resp, "text", None)
        except Exception:
            pass
        try:
            if hasattr(resp, "json"):
                details["response_json"] = resp.json()
        except Exception:
            pass

    # Some SDKs store raw payload in .data / .error
    for attr in ("data", "error"):
        try:
            if hasattr(err, attr):
                details[attr] = getattr(err, attr)
        except Exception:
            pass

    try:
        return json.dumps(details, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(details)


def _effective_model_for_provider(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider == "google":
        return (
            os.getenv("PLOTEXTRACT_GOOGLE_MODEL")
            or (LLM_MODEL if LLM_MODEL else None)
            or "gemma-3-27b-it"
        )
    # mistral (default)
    return (
        os.getenv("PLOTEXTRACT_MISTRAL_MODEL")
        or (LLM_MODEL if LLM_MODEL else None)
        or "mistral-large-2512"
    )


def _effective_key_for_provider(provider: str, key: str) -> str:
    key = (key or "").strip()
    if key:
        return key
    provider = (provider or "").strip().lower()
    if provider == "google":
        return "2"
    if provider == "mistral":
        return "4"
    return ""


def _select_mistral_api_key(llm_key_used: str) -> Optional[str]:
    """Select the Mistral API key based on the requested key slot.

    Conventions:
    - key1  -> API_KEY_1
    - key3  -> API_KEY_3
    - key4  -> API_KEY_4 (preferred), else API_KEY_3, else API_KEY_1
    """
    k = (llm_key_used or "").strip()
    if k == "1":
        return API_KEY_MISTRAL_1
    if k == "3":
        return API_KEY_MISTRAL_3 or API_KEY_MISTRAL_1
    if k == "4":
        return API_KEY_MISTRAL_4 or API_KEY_MISTRAL_3 or API_KEY_MISTRAL_1
    # Unknown/empty: best-effort fallback
    return API_KEY_MISTRAL_4 or API_KEY_MISTRAL_3 or API_KEY_MISTRAL_1

# Debug mode (writes per-stage prompts/outputs to disk)
DEBUG_VERBOSE = str(os.getenv("PLOTEXTRACT_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}

if len(sys.argv) < 3:
    print("Usage: python plot_extract_v2/runner.py <path_to_plot_image> <prompt_name> [article_info]\nError: Missing required argument. Please provide the path to the plot image and the prompt name (e.g., prompt_1).")
    sys.exit(1)

input_plot = sys.argv[1]
prompt_name = sys.argv[2]
article_info_text = sys.argv[3] if len(sys.argv) > 3 else ""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_module(module_path):
    spec = importlib.util.spec_from_file_location("dynamic_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def save_stage_update(output_dir, stage_name, stage_index, total_stages, accumulated_facts, stage_time_ms, console_output=""):
    """Save real-time stage update for the web UI to poll.

    Notes on semantics:
    - `stage_index` represents the number of *completed* stages (0..total_stages).
    - `stage` represents the current stage name (or the most recently completed stage).
    - Progress percentage is derived only from completed stages.
    """
    try:
        # Determine if this is a completion update.
        # IMPORTANT: Do not treat the last stage as "complete" because the pipeline still
        # performs post-processing after the stage loop (CSV finalization, replot, comparisons).
        is_complete = stage_name == "COMPLETE"
        
        completed_stages = max(0, min(int(stage_index), int(total_stages))) if total_stages is not None else 0
        percentage = round((completed_stages / total_stages) * 100, 1) if total_stages and total_stages > 0 else 0

        update_data = {
            "status": "complete" if is_complete else "running",
            "stage": stage_name,
            "stage_index": completed_stages,
            "total_stages": total_stages,
            "percentage": 100.0 if is_complete else percentage,
            "accumulated_facts": accumulated_facts,
            "stage_duration_ms": stage_time_ms,
            "timestamp": time.time(),
            "console_output": console_output
        }
        
        update_file = os.path.join(output_dir, "_extraction_progress.json")
        with open(update_file, "w", encoding="utf-8") as f:
            json.dump(update_data, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save stage update: {e}")


def _snapshot_jsonable(obj):
    """Best-effort immutable snapshot for console logging.

    Prefer a JSON round-trip (guarantees no shared references); fall back to deepcopy.
    """
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False))
    except Exception:
        try:
            return copy.deepcopy(obj)
        except Exception:
            return obj


def _format_stage_facts_dump(stage_number: int, accumulated_facts: dict) -> str:
    header = f"===== STAGE {stage_number} COMPLETE — ACCUMULATED FACTS ====="
    snapshot = _snapshot_jsonable(accumulated_facts)
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False, default=str)
    return f"{header}\n{payload}\n"


def _safe_console_print(text: str) -> None:
    """Print text robustly even when terminal encoding cannot represent all chars."""
    try:
        print(text)
        return
    except UnicodeEncodeError:
        pass

    enc = (getattr(sys.stdout, "encoding", None) or "utf-8")
    safe_text = str(text).encode(enc, errors="backslashreplace").decode(enc, errors="replace")
    print(safe_text)


def _extract_json_object_from_text(text: str):
    """Best-effort JSON extraction.

    The model sometimes wraps JSON in ```json fences, or includes extra text.
    This function tries (in order): direct parse, fenced-block parse, and
    a substring from first '{' to last '}'. Returns the parsed object or None.
    """
    if not text:
        return None

    raw = text.strip()

    # 1) Direct parse (fast path)
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) If the whole content is fenced, strip the outer fence
    if raw.startswith("```"):
        unfenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        unfenced = re.sub(r"\s*```$", "", unfenced)
        try:
            return json.loads(unfenced.strip())
        except Exception:
            pass

    # 3) Parse first fenced JSON block
    for match in re.finditer(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE):
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # 4) Parse best-effort substring from first '{'..last '}'
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None

def load_prompt_set(prompt_name):
    """Load prompts from prompt_name/prompts.py and chain metadata."""
    prompt_set_dir = os.path.join(PROMPTS_DIR, prompt_name)
    
    if not os.path.exists(prompt_set_dir):
        raise FileNotFoundError(f"Prompt set '{prompt_name}' not found at {prompt_set_dir}")
    
    # Load prompts from prompt_name/prompts.py
    prompts_file = os.path.join(prompt_set_dir, 'prompts.py')
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"Prompts file not found at {prompts_file}")
    
    prompts_module = load_module(prompts_file)
    
    # Load chain metadata from chains/chain_<prompt_name>.py
    chains_dir = os.path.join(PROMPTS_DIR, 'chains')
    chain_file = os.path.join(chains_dir, f'chain_{prompt_name}.py')
    
    if not os.path.exists(chain_file):
        raise FileNotFoundError(f"Chain metadata file not found at {chain_file}")
    
    chain_module = load_module(chain_file)
    
    # Validate chain has required attributes
    if not hasattr(chain_module, 'EXTRACT_STAGES'):
        raise ValueError(f"Chain file {chain_file} missing 'EXTRACT_STAGES' list")
    
    return prompts_module, chain_module


def load_extraction_schema():
    """Load the complete extraction schema."""
    schema_path = os.path.join(PROMPTS_DIR, 'extraction_schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Extraction schema not found at {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_abort_signal(result_text):
    """Detect whether a stage signaled an abort.

    Accepts multiple formats for backward compatibility:
    - Exact string "None" (legacy early-stop)
    - A response starting with "ABORT" (free-form abort message)
    - JSON object containing {"abort": true, "reason": "..."}
      or nested abort flags under known sections (e.g., marker_facts, axis_facts).
    Returns (abort: bool, reason: Optional[str], confidence: Optional[float]).
    """
    text = str(result_text).strip()

    # Legacy: exact None
    if text == "None":
        return True, "Legacy abort: 'None'", None

    # Free-form: starts with ABORT
    if text.upper().startswith("ABORT"):
        return True, text, None

    # Structured: JSON with abort flag
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            # Top-level abort
            if obj.get("abort") is True:
                reason = obj.get("reason") or obj.get("abort_reason") or "Abort signaled by prompt"
                conf = obj.get("confidence")
                try:
                    conf = float(conf) if conf is not None else None
                except (TypeError, ValueError):
                    conf = None
                return True, reason, conf

            # Nested abort under known sections
            for section in ("marker_facts", "axis_facts"):
                sec = obj.get(section)
                if isinstance(sec, dict) and sec.get("abort") is True:
                    reason = sec.get("reason") or "Abort signaled by prompt"
                    conf = sec.get("confidence")
                    try:
                        conf = float(conf) if conf is not None else None
                    except (TypeError, ValueError):
                        conf = None
                    return True, reason, conf
    except (json.JSONDecodeError, TypeError):
        pass

    return False, None, None


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_Q_1p(convo):
    Q = []
    for ic, c in enumerate(convo):
        role = "user" if ic % 2 == 0 else "assistant"
        if isinstance(c, list):
            Q.append({
                "role": role,
                "content": [
                    {"type": "text", "text": c[1]},
                    {"type": "image_url", "image_url": {"url": f"data:image/{PNGJPG};base64,{c[0]}"}},
                ],
            })
        else:
            Q.append({"role": role, "content": c})
    return Q


def prompt_mistral(client, messages):
    # Prevent indefinite hangs (e.g. network stalls / provider queueing).
    # Override per run with env var PLOTEXTRACT_MISTRAL_TIMEOUT_MS.
    # Default: 4 minutes. Set PLOTEXTRACT_MISTRAL_TIMEOUT_MS=0 to disable.
    timeout_ms: Optional[int] = None
    try:
        timeout_ms_env = os.getenv("PLOTEXTRACT_MISTRAL_TIMEOUT_MS")
        if timeout_ms_env is not None and str(timeout_ms_env).strip() != "":
            v = int(str(timeout_ms_env).strip())
            timeout_ms = None if v <= 0 else v
        else:
            timeout_ms = 240_000
    except Exception:
        timeout_ms = 240_000

    # Allow overriding model via env var.
    # Default remains the legacy value to avoid breaking existing pipelines.
    model = (
        os.getenv("PLOTEXTRACT_MISTRAL_MODEL")
        or (LLM_MODEL if LLM_MODEL else None)
        or "mistral-large-2512"
    )

    def _messages_contain_images(msgs) -> bool:
        try:
            for m in (msgs or []):
                if not isinstance(m, dict):
                    continue
                content = m.get("content")
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "image_url":
                            return True
        except Exception:
            return False
        return False

    # Some Mistral models are text-only; if the prompt includes images, fall back.
    # This keeps the UI option available without breaking image-based extraction.
    if model == "mistral-large-2411" and _messages_contain_images(messages):
        fallback_model = os.getenv("PLOTEXTRACT_MISTRAL_VISION_FALLBACK_MODEL") or "mistral-large-2512"
        print(
            f"[WARN] Selected model '{model}' does not support image input; falling back to '{fallback_model}'.",
            file=sys.stderr,
        )
        model = fallback_model

    def _is_rate_limit_error(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "status 429" in msg
            or "rate limit" in msg
            or "rate_limited" in msg
            or '"code":"1300"' in msg
            or "too many requests" in msg
        )

    # Retry transient rate limits (429).
    # Defaults are intentionally more forgiving than before: rate-limit windows
    # can easily be 60s+ and EX2 runs multiple LLM calls per extraction.
    try:
        max_retries = int(os.getenv("PLOTEXTRACT_MISTRAL_MAX_RETRIES") or "8")
    except Exception:
        max_retries = 8
    try:
        base_sleep_s = float(os.getenv("PLOTEXTRACT_MISTRAL_RETRY_BASE_S") or "2.0")
    except Exception:
        base_sleep_s = 2.0
    try:
        max_sleep_s = float(os.getenv("PLOTEXTRACT_MISTRAL_RETRY_MAX_S") or "45.0")
    except Exception:
        max_sleep_s = 45.0

    # Add some jitter to avoid synchronized retries when multiple runs are started.
    try:
        jitter_frac = float(os.getenv("PLOTEXTRACT_MISTRAL_RETRY_JITTER_FRAC") or "0.20")
    except Exception:
        jitter_frac = 0.20
    jitter_frac = max(0.0, min(jitter_frac, 0.9))

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            _maybe_throttle_llm_call()
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0,
            }
            if timeout_ms is not None:
                kwargs["timeout_ms"] = timeout_ms
            response = client.chat.complete(
                **kwargs,
            )
            return messages, response.choices[0].message.content
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e):
                # Log the full SDK error payload at least once.
                if attempt == 0:
                    print(
                        "[WARN] Mistral rate limit details (first 429):\n" + _format_llm_exception_details(e),
                        file=sys.stderr,
                    )
                if attempt < max_retries:
                    sleep_s = min(max_sleep_s, base_sleep_s * (2 ** attempt))
                    if jitter_frac > 0:
                        sleep_s = sleep_s * (1.0 + (random.random() * jitter_frac))
                    print(
                        f"[WARN] Mistral rate limit (429). Retrying in {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})",
                        file=sys.stderr,
                    )
                    time.sleep(sleep_s)
                    continue
                # Last attempt exhausted.
                print(
                    "[ERROR] Mistral rate limit: retries exhausted. Last error details:\n"
                    + _format_llm_exception_details(e),
                    file=sys.stderr,
                )
            raise

    # Should be unreachable, but keep for safety.
    if last_err is not None:
        raise last_err
    raise RuntimeError("prompt_mistral failed without an exception")


def prompt_google(messages):
    """Send a prompt to Google GenAI (Gemini/Gemma via API key).

    This is a best-effort adapter that accepts the same `messages` structure used
    by the Mistral path: a list of {role, content}, where content can be a string
    or a list of typed parts (text + image_url data URIs).
    """
    if not API_KEY_GOOGLE:
        raise RuntimeError("Missing API_KEY_2 for Google provider")

    try:
        import google.generativeai as genai  # type: ignore
        from google.generativeai import types as genai_types  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Google provider requested but 'google-generativeai' is not installed. "
            "Install it with: pip install google-generativeai"
        ) from e

    genai.configure(api_key=API_KEY_GOOGLE)

    model_name = (
        os.getenv("PLOTEXTRACT_GOOGLE_MODEL")
        or (LLM_MODEL if LLM_MODEL else None)
        # Default requested by user. If your Google endpoint expects a different
        # model id for "Gemma 3 27B Vision", set PLOTEXTRACT_GOOGLE_MODEL in .env.
        or "gemma-3-27b-it"
    )

    def _is_rate_limit_error(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            "429" in msg
            or "rate limit" in msg
            or "too many requests" in msg
            or "resource exhausted" in msg
        )

    try:
        max_retries = int(os.getenv("PLOTEXTRACT_GOOGLE_MAX_RETRIES") or "10")
    except Exception:
        max_retries = 10
    try:
        base_sleep_s = float(os.getenv("PLOTEXTRACT_GOOGLE_RETRY_BASE_S") or "2.0")
    except Exception:
        base_sleep_s = 2.0
    try:
        max_sleep_s = float(os.getenv("PLOTEXTRACT_GOOGLE_RETRY_MAX_S") or "60.0")
    except Exception:
        max_sleep_s = 60.0

    try:
        jitter_frac = float(os.getenv("PLOTEXTRACT_GOOGLE_RETRY_JITTER_FRAC") or "0.20")
    except Exception:
        jitter_frac = 0.20
    jitter_frac = max(0.0, min(jitter_frac, 0.9))

    # Flatten messages into a single prompt string plus binary image parts.
    prompt_lines = []
    parts = []

    def _add_text(t: str):
        if t is None:
            return
        t = str(t)
        if not t.strip():
            return
        prompt_lines.append(t)

    for m in messages or []:
        role = (m.get("role") or "user").upper()
        content = m.get("content")
        if isinstance(content, str):
            _add_text(f"{role}: {content}")
            continue

        # Typed parts (text + image_url)
        if isinstance(content, list):
            # Collect any text first
            text_chunks = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    text_chunks.append(p.get("text") or "")
            if text_chunks:
                _add_text(f"{role}: " + "\n".join([t for t in text_chunks if str(t).strip()]))

            # Then images
            for p in content:
                if not isinstance(p, dict) or p.get("type") != "image_url":
                    continue
                url = (((p.get("image_url") or {}) if isinstance(p.get("image_url"), dict) else {}) or {}).get("url")
                if not url or not isinstance(url, str):
                    continue
                if url.startswith("data:") and ";base64," in url:
                    header, b64 = url.split(",", 1)
                    mime = header.split(";", 1)[0].replace("data:", "") or "image/png"
                    try:
                        img_bytes = base64.b64decode(b64)
                    except Exception:
                        continue

                    # Prefer official Blob type when available.
                    try:
                        parts.append(genai_types.Blob(mime_type=mime, data=img_bytes))
                    except Exception:
                        parts.append({"mime_type": mime, "data": img_bytes})
                else:
                    # Non-data URI not supported in this adapter
                    continue

            continue

        # Unknown content type
        _add_text(f"{role}: {str(content)}")

    prompt_text = "\n\n".join(prompt_lines).strip()
    if prompt_text:
        parts.insert(0, prompt_text)

    model = genai.GenerativeModel(model_name=model_name)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            _maybe_throttle_llm_call()
            resp = model.generate_content(
                parts,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 4096,
                },
            )
            text = getattr(resp, "text", None)
            if not text:
                # Try to extract from candidates/parts
                try:
                    c0 = resp.candidates[0]
                    text = c0.content.parts[0].text
                except Exception:
                    text = ""
            return messages, str(text)
        except Exception as e:
            last_err = e
            if _is_rate_limit_error(e) and attempt < max_retries:
                sleep_s = min(max_sleep_s, base_sleep_s * (2 ** attempt))
                if jitter_frac > 0:
                    sleep_s = sleep_s * (1.0 + (random.random() * jitter_frac))
                print(
                    f"[WARN] Google rate limit (429). Retrying in {sleep_s:.1f}s (attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
                continue
            raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("prompt_google failed without an exception")


def prompt_llm(mistral_client, messages):
    """Provider switch for LLM prompting."""
    provider = (LLM_PROVIDER or "mistral").strip().lower()
    if provider == "google":
        return prompt_google(messages)

    if provider != "mistral":
        print(f"[WARN] Unknown LLM provider '{provider}', falling back to mistral", file=sys.stderr)

    if mistral_client is None:
        raise RuntimeError("Mistral provider requested but client is not available")
    return prompt_mistral(mistral_client, messages)


def clean_code_response(code_text):
    """Remove markdown code fences and other formatting from LLM response."""
    code_text = code_text.strip()
    
    # Remove markdown code fences
    if code_text.startswith("```python"):
        code_text = code_text[len("```python"):].strip()
    elif code_text.startswith("```"):
        code_text = code_text[3:].strip()
    
    if code_text.endswith("```"):
        code_text = code_text[:-3].strip()
    
    return code_text


def extract_csv_from_text(text):
    """Extract pure CSV content from mixed LLM output (csv fenced block or JSON marker_facts)."""
    text = text.strip()

    # Try direct JSON parsing first (for complete JSON responses)
    try:
        obj = json.loads(text)
        csv_val = obj.get("marker_facts", {}).get("csv_output")
        if csv_val:
            # Unescape if needed
            if isinstance(csv_val, str):
                csv_val = csv_val.replace("\\n", "\n").replace("\\\\", "\\")
            return csv_val.strip()
    except (json.JSONDecodeError, TypeError):
        pass

    # Try JSON block with marker_facts.csv_output (for mixed content)
    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.S)
    if json_match:
        try:
            obj = json.loads(json_match.group(1))
            csv_val = obj.get("marker_facts", {}).get("csv_output")
            if csv_val:
                if isinstance(csv_val, str):
                    csv_val = csv_val.replace("\\n", "\n").replace("\\\\", "\\")
                return csv_val.strip()
        except Exception:
            pass

    # Try fenced CSV block
    csv_match = re.search(r"```csv\s*(.*?)\s*```", text, re.S)
    if csv_match:
        return csv_match.group(1).strip()

    # Try to find CSV-like content by looking for lines with numbers and commas
    # This is a fallback for when the LLM output is not properly formatted
    lines = text.split('\n')
    csv_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and non-CSV-like content
        if not line or line.startswith("Here is") or line.startswith("Step "):
            continue
        # Look for lines that look like CSV (contain commas and numbers/text)
        if ',' in line:
            csv_lines.append(line)
    
    # If we found CSV-like lines, return them
    if csv_lines and len(csv_lines) > 1:  # At least header + 1 data row
        return "\n".join(csv_lines)

    # Fallback: return raw text
    return text


def rebuild_csv_from_json_curves(accumulated_facts):
    """Rebuild clean CSV from marker_facts.curves JSON, ignoring the LLM's csv_output string.
    
    This ensures we use the actual structured points rather than the LLM's often-malformed CSV."""
    try:
        marker_facts = accumulated_facts.get("marker_facts", {})
        curves = marker_facts.get("curves", [])
        if not curves:
            return None
        
        # Collect all unique x values and curve labels
        x_set = {}  # {x_value: index} to preserve order
        curve_labels = []
        seen_curves = set()
        
        for curve_obj in curves:
            label = curve_obj.get("curve_label", "Unknown")
            if label not in seen_curves:
                curve_labels.append(label)
                seen_curves.add(label)
            
            points = curve_obj.get("points", [])
            for pt in points:
                x = pt.get("x")
                if x is not None and x not in x_set:
                    x_set[x] = len(x_set)
        
        if not x_set or not curve_labels:
            return None
        
        x_values = sorted(x_set.keys(), key=lambda v: x_set[v])
        
        # Build mapping (x, curve_label) -> y
        y_map = {}
        for curve_obj in curves:
            label = curve_obj.get("curve_label")
            points = curve_obj.get("points", [])
            for pt in points:
                x = pt.get("x")
                y = pt.get("y")
                if x is not None and y is not None:
                    y_map[(x, label)] = y
        
        # Build header: Time (hours), log10 CFU/mL (Curve1), Time (hours), log10 CFU/mL (Curve2), ...
        header = []
        for label in curve_labels:
            header.append("Time (hours)")
            header.append(f"log10 CFU/mL ({label})")
        
        lines = [",".join(header)]
        
        # Build data rows
        for x in x_values:
            row = []
            for label in curve_labels:
                row.append(str(x))
                y = y_map.get((x, label), "")
                row.append(str(y) if y != "" else "")
            lines.append(",".join(row))
        
        return "\n".join(lines)
    except Exception as e:
        print(f"Error rebuilding CSV from JSON: {e}")
        return None


def _result_has_marker_facts(result_json) -> bool:
    if not isinstance(result_json, dict):
        return False
    marker_facts = result_json.get("marker_facts")
    return isinstance(marker_facts, dict)


def normalize_csv_to_wide(csv_text: str) -> str:
    """Convert long-form CSV (x, curve_label, y) into wide format expected by interpolation.

    If the CSV is already wide (multiple x,y column pairs), it is returned unchanged.
    """
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return csv_text

    header = [h.strip() for h in lines[0].split(",")]

    # Check if already in wide format (even number of columns with alternating x/y pairs)
    # Wide format: "Time (hours)", "CFU/mL (Condition 1)" [2 cols for 1 curve]
    #             OR "Time (hours)", "CFU/mL (Condition 1)", "Time (hours)", "CFU/mL (Condition 2)" [4+ cols for 2+ curves]
    if len(header) >= 2 and len(header) % 2 == 0:
        # Check if columns alternate between x and y descriptors
        is_wide = True
        for i in range(0, len(header), 2):
            x_col = header[i].lower()
            # X column should contain "time" or "hour" or similar
            if not any(kw in x_col for kw in ["time", "hour", "x"]):
                is_wide = False
                break
        if is_wide:
            return csv_text  # Already wide format, return unchanged

    # Detect long format variants
    # Case A: strict 3-column long format: x, curve_label, y
    # Case B: 4+ columns where last column is condition/curve label and one numeric column is y
    rows = []
    if len(header) == 3:
        x_col, label_col, y_col = header
        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) != 3:
                continue
            x_val, curve_label, y_val = parts
            rows.append((x_val, curve_label, y_val))
    else:
        # Try heuristic: last column is curve label (condition), first column is x, one numeric column is y
        # Adjust for cases where header has more columns than data rows
        label_idx = len(header) - 1
        x_idx = 0

        # Determine the max actual column count across data rows
        max_cols = 0
        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) > max_cols:
                max_cols = len(parts)
        if max_cols == 0:
            return csv_text

        # Clamp label_idx to existing columns
        if label_idx >= max_cols:
            label_idx = max_cols - 1

        # pick y_idx as last numeric-ish column before label
        y_idx = None
        for idx in range(min(label_idx - 1, max_cols - 1), -1, -1):
            sample_val = None
            for ln in lines[1:]:
                parts = [p.strip() for p in ln.split(",") if p.strip()]
                if len(parts) > idx:
                    sample_val = parts[idx]
                    break
            if sample_val is not None:
                try:
                    float(sample_val)
                    y_idx = idx
                    break
                except ValueError:
                    continue
        if y_idx is None:
            return csv_text  # cannot detect

        x_col = header[x_idx]
        y_col = header[y_idx]
        label_col = header[label_idx]

        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) <= max(x_idx, y_idx, label_idx):
                continue
            x_val = parts[x_idx]
            y_val = parts[y_idx]
            curve_label = parts[label_idx]
            rows.append((x_val, curve_label, y_val))

    if not rows:
        return csv_text

    # Collect unique x values in original order and curve labels
    x_values = []
    curves = []
    seen_x = set()
    seen_curve = set()
    for x_val, curve_label, _ in rows:
        if x_val not in seen_x:
            seen_x.add(x_val)
            x_values.append(x_val)
        if curve_label not in seen_curve:
            seen_curve.add(curve_label)
            curves.append(curve_label)

    # Build mapping: (x, curve) -> y
    y_map = {}
    for x_val, curve_label, y_val in rows:
        y_map[(x_val, curve_label)] = y_val

    # Construct wide header: Time, Y(curve1), Time, Y(curve2), ...
    wide_header = []
    for curve in curves:
        wide_header.append(x_col)
        wide_header.append(f"{y_col} ({curve})")

    wide_lines = [",".join(wide_header)]

    for x_val in x_values:
        row_vals = []
        for curve in curves:
            row_vals.append(x_val)
            row_vals.append(y_map.get((x_val, curve), ""))
        wide_lines.append(",".join(row_vals))

    return "\n".join(wide_lines)


def _csv_has_usable_curve_points(csv_text: str, min_points_per_curve: int = 2) -> bool:
    """Return True when CSV appears to contain usable extracted curve point series.

    We normalize to wide format and run deterministic diagnostics. This guards
    against non-curve tables (e.g., review/checklist CSVs) being treated as
    extracted time-series data.
    """
    try:
        text = str(csv_text or "").strip()
        if not text:
            return False

        normalized = normalize_csv_to_wide(text)
        diag = compute_csv_diagnostics_from_text(normalized)
        curves = diag.get("curves", []) if isinstance(diag, dict) else []
        if not isinstance(curves, list) or not curves:
            return False

        for c in curves:
            if not isinstance(c, dict):
                continue
            n_points = c.get("n_points")
            try:
                if int(n_points) >= int(min_points_per_curve):
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _extract_usable_csv_from_stage_text(stage_text: str) -> str:
    """Extract and normalize CSV from stage text, returning empty string when unusable."""
    candidate = normalize_csv_to_wide(extract_csv_from_text(stage_text or "") or "")
    if _csv_has_usable_curve_points(candidate):
        return candidate
    return ""


def _find_best_prior_csv(stage_context: dict, stage_order: list[str], current_stage_index: int) -> tuple[str, str | None]:
    """Find the nearest prior stage that contains usable curve CSV."""
    for idx in range(int(current_stage_index) - 1, -1, -1):
        stage_name = stage_order[idx]
        stage_text = stage_context.get(stage_name, "")
        if not stage_text:
            continue
        candidate = _extract_usable_csv_from_stage_text(stage_text)
        if candidate:
            return candidate, stage_name
    return "", None


def _suffix_from_tags(*tags: str | None) -> str:
    parts: list[str] = []
    for t in tags:
        if not t:
            continue
        t = str(t).strip()
        if not t:
            continue
        if t.startswith('.'):
            t = t[1:]
        if not t:
            continue
        parts.append(t)
    return ''.join(f".{p}" for p in parts)


def get_next_version(parent_dir, name_for_folder, prompt_name, key_tag: str | None = None, output_tag: str | None = None):
    """Return the next version number and folder path.

    Important: version numbers are monotonic across keys.
    i.e. if v4 exists (with or without .keyN), next will be v5.<key_tag>.
    """
    import re

    suffix = _suffix_from_tags(key_tag, output_tag)

    max_version = 0
    try:
        pat = re.compile(
            rf'^{re.escape(name_for_folder)}\.{re.escape(prompt_name)}\.v(\d+)(?:\.key\d+)?(?:\.web)?$'
        )
        if os.path.isdir(parent_dir):
            for item in os.listdir(parent_dir):
                m = pat.match(item)
                if not m:
                    continue
                try:
                    v = int(m.group(1))
                except Exception:
                    continue
                if v > max_version:
                    max_version = v
    except Exception:
        max_version = 0

    version = max_version + 1
    folder_name = f"{name_for_folder}.{prompt_name}.v{version}{suffix}"
    folder_path = os.path.join(parent_dir, folder_name)
    return version, folder_path


def stack_images_vertically(image1_path, image2_path, border_color, output_dir, prompt_name, version_num, key_tag: str | None = None, output_tag: str | None = None, border_size=30):
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None or img2 is None:
        print(f"Error: Cannot read images. img1={image1_path}, img2={image2_path}")
        return None

    width = max(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
    img2_resized = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))

    label_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    font_thickness = 3
    label1 = np.ones((label_height, width, 3), dtype=np.uint8) * 255
    label2 = np.ones((label_height, width, 3), dtype=np.uint8) * 255
    cv2.putText(label1, "Original", (15, 45), font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(label2, "Extracted (Re-plotted)", (15, 45), font, font_scale, (0, 0, 0), font_thickness)
    combined_image = np.vstack((label1, img1_resized, label2, img2_resized))

    if "yes" in border_color.lower():
        color = (0, 255, 0)
    elif "no" in border_color.lower():
        color = (0, 0, 255)
    else:
        print("Invalid border color input. Use 'yes' for green or 'no' for red.")
        return None

    combined_image_with_border = cv2.copyMakeBorder(
        combined_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    # Validation images can exceed the provider's multimodal limits when stacking.
    # Downscale to a safe maximum dimension (default 1024) to avoid HTTP 400 errors.
    try:
        max_dim = int(os.getenv("PLOTEXTRACT_VALIDATION_MAX_DIM", "1024"))
    except Exception:
        max_dim = 1024
    if max_dim and max_dim > 0:
        h, w = combined_image_with_border.shape[:2]
        max_wh = max(w, h)
        if max_wh > max_dim:
            scale = max_dim / float(max_wh)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            combined_image_with_border = cv2.resize(combined_image_with_border, (new_w, new_h), interpolation=cv2.INTER_AREA)

    original_filename = os.path.basename(image1_path)
    base_name = original_filename.rsplit(".", 1)[0] if "." in original_filename else original_filename
    suffix = _suffix_from_tags(key_tag, output_tag)
    output_filename = os.path.join(output_dir, f"comparison_{base_name}.{prompt_name}.v{version_num}{suffix}.png")
    cv2.imwrite(output_filename, combined_image_with_border)
    return output_filename


# NOTE: Prompts and stages are loaded via load_prompt_set() function after image encoding below

# Load prompts and chain metadata from the new modular structure
try:
    prompts_module, chain_module = load_prompt_set(prompt_name)
except (FileNotFoundError, ValueError) as e:
    print(f"Error loading prompt set: {e}")
    sys.exit(1)

CHAIN_NAME = getattr(chain_module, "CHAIN_NAME", prompt_name)
EXTRACT_STAGES = chain_module.EXTRACT_STAGES
COMPLETE_SCHEMA = getattr(chain_module, "COMPLETE_SCHEMA", ACCUMULATED_FACTS_SCHEMA)
COMPLETE_SCHEMA_CONSTRAINTS = getattr(chain_module, "COMPLETE_SCHEMA_CONSTRAINTS", SCHEMA_CONSTRAINTS)
NO_IMAGE_STAGES = set(getattr(chain_module, "NO_IMAGE_STAGES", []))

# Verify all stages exist in prompts module
for stage_name in EXTRACT_STAGES:
    if not hasattr(prompts_module, stage_name):
        print(f"Error: Stage '{stage_name}' not found in prompts module")
        sys.exit(1)

# Also load shared prompts (CODE_PLOT, CODE_FIX, COMPARE_*)
for prompt_attr in ['CODE_PLOT', 'CODE_FIX', 'COMPARE_X', 'COMPARE_Y', 'COMPARE_NUMBER', 'COMPARE_TREND']:
    if not hasattr(prompts_module, prompt_attr):
        print(f"Warning: {prompt_attr} not found in prompts module")

# Build a version identifier using the prompt name
prompt_version = f"pv2_{prompt_name}"

# -----------------------------------------------------------------------------
# Image handling
# -----------------------------------------------------------------------------
original_ext = os.path.splitext(input_plot)[-1].lower()
api_image_path = input_plot
PNGJPG = original_ext.lstrip(".")
if PNGJPG == "jpg":
    PNGJPG = "jpeg"
elif PNGJPG == "svg":
    import subprocess as sp
    import tempfile

    temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    abs_svg = os.path.abspath(input_plot).replace("\\", "/")
    html_content = f"""<!DOCTYPE html>
<html><head><style>
html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: white; }}
img {{ width: 100%; height: 100%; object-fit: contain; }}
</style></head>
<body><img src=\"file:///{abs_svg}\"></body></html>"""

    temp_html = tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8")
    temp_html.write(html_content)
    temp_html.close()

    browsers = [
        r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
        r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
        "msedge",
        r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
        r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        "chrome",
    ]

    converted = False
    for browser in browsers:
        try:
            sp.run(
                [
                    browser,
                    "--headless",
                    "--disable-gpu",
                    f"--screenshot={temp_png}",
                    "--window-size=1600,1200",
                    "--force-device-scale-factor=2",
                    f"file:///{temp_html.name.replace(os.sep, '/')}",
                ],
                capture_output=True,
                timeout=60,
            )
            if os.path.exists(temp_png) and os.path.getsize(temp_png) > 0:
                api_image_path = temp_png
                PNGJPG = "png"
                converted = True
                print("Converted SVG to PNG (high resolution)")
                break
        except (FileNotFoundError, sp.TimeoutExpired, OSError):
            continue

    try:
        os.unlink(temp_html.name)
    except Exception:
        pass

    if not converted:
        print("ERROR: Cannot convert SVG to PNG. Install Edge or Chrome.")
        sys.exit(1)

base64_image = encode_image(api_image_path)

# Initialize provider client(s)
client = None
if LLM_PROVIDER == "mistral":
    llm_key_used = _effective_key_for_provider("mistral", LLM_KEY)
    api_key_mistral = _select_mistral_api_key(llm_key_used)
    if not api_key_mistral:
        raise RuntimeError(
            "Missing Mistral API key for selected key slot. "
            "Set API_KEY_4 (preferred for key4), or API_KEY_3, or API_KEY_1."
        )
    if Mistral is None:
        raise RuntimeError("Mistral provider requested but 'mistralai' is not installed")
    if llm_key_used == "4" and (not API_KEY_MISTRAL_4) and API_KEY_MISTRAL_3:
        print("[WARN] key4 selected but API_KEY_4 is not set; using API_KEY_3 instead.", file=sys.stderr)
    client = Mistral(api_key=api_key_mistral)

# -----------------------------------------------------------------------------
# Preprocessing: Convert article info to structured JSON (if provided)
# -----------------------------------------------------------------------------
ARTICLE_INFO_PROMPT = r"""You are given free-text information copied from a research article that describes a plot (e.g. figure caption, legend text, brief experimental context).

Your task is to convert this text into a structured JSON object called `article_info`, following the schema below.

Rules (non-negotiable):
- Use ONLY the information explicitly present in the provided text.
- Do NOT guess, infer, or fill in missing fields.
- If a field cannot be confidently populated, OMIT it entirely.
- Do NOT paraphrase figure captions or legend text; copy verbatim where applicable.
- Output VALID JSON only. No explanations, no commentary, no extra text.

The resulting JSON will be treated as hard facts and passed unchanged to later extraction stages.

JSON schema to follow:

{schema}

{constraints}

Input:
{article_text}

Output:
(valid JSON only)"""

article_info_json = ""
if article_info_text:
    print("Preprocessing article information...")
    article_prompt = ARTICLE_INFO_PROMPT.format(
        schema=COMPLETE_SCHEMA,
        constraints=COMPLETE_SCHEMA_CONSTRAINTS,
        article_text=article_info_text
    )
    messages = [{"role": "user", "content": article_prompt}]
    _, article_response = prompt_llm(client, messages)
    
    # Clean response and validate JSON
    article_response = article_response.strip()
    if article_response.startswith("```json"):
        article_response = article_response[len("```json"):].strip()
    if article_response.startswith("```"):
        article_response = article_response[3:].strip()
    if article_response.endswith("```"):
        article_response = article_response[:-3].strip()
    
    try:
        json.loads(article_response)  # Validate JSON
        article_info_json = article_response
        print("Article information structured successfully")
    except json.JSONDecodeError:
        print("Warning: Article information could not be parsed as valid JSON, skipping")
        article_info_json = ""

# -----------------------------------------------------------------------------
# Output naming
# -----------------------------------------------------------------------------
image_filename = os.path.basename(input_plot)
base_name = os.path.splitext(image_filename)[0]
name_for_folder = image_filename.replace(".", "_")
full_prompt_name = f"pv2_{prompt_name}"
llm_provider_used = (LLM_PROVIDER or "mistral").strip().lower()
llm_key_used = _effective_key_for_provider(llm_provider_used, LLM_KEY)
key_tag = f"key{llm_key_used}" if llm_key_used else None
try:
    output_tag = str(os.getenv("PLOTEXTRACT_OUTPUT_TAG", "")).strip()
except Exception:
    output_tag = ""
if output_tag.startswith('.'):
    output_tag = output_tag[1:]
if output_tag.lower() == '':
    output_tag = ""
if output_tag and output_tag.lower() != 'web':
    # For now we only support a single known tag to avoid creating accidental new naming schemes.
    output_tag = ""

version_num, version_dir = get_next_version(
    os.path.dirname(input_plot),
    name_for_folder,
    full_prompt_name,
    key_tag=key_tag,
    output_tag=(output_tag or None),
)
os.makedirs(version_dir, exist_ok=True)

debug_dir = os.path.join(version_dir, "debug")
if DEBUG_VERBOSE:
    os.makedirs(debug_dir, exist_ok=True)

run_suffix = _suffix_from_tags(key_tag, output_tag)
output_out = os.path.join(version_dir, f"{image_filename}.{full_prompt_name}.v{version_num}{run_suffix}.mistral.out")
replot_plot = os.path.join(version_dir, f"{name_for_folder}-replot.{full_prompt_name}.v{version_num}{run_suffix}.png")

print(f"Input plot: {input_plot}")
print(f"Using prompt set: {prompt_name}")
print(f"Output folder: {version_dir} (version {version_num})")

# Record which LLM/key/model were used for this run.
try:
    llm_model_used = _effective_model_for_provider(llm_provider_used)
    llm_info = {
        "llm_provider": llm_provider_used,
        "llm_model": llm_model_used,
        "llm_key": (f"key{llm_key_used}" if llm_key_used else ""),
        "llm_key_index": llm_key_used,
    }
    with open(os.path.join(version_dir, "llm_info.json"), "w", encoding="utf-8") as f:
        json.dump(llm_info, f, indent=2)
    print(
        f"LLM: provider={llm_provider_used} model={llm_model_used} key={llm_info['llm_key']}",
        file=sys.stderr,
    )
except Exception:
    pass

stage_context = {}
conversation_log = []
console_timeline = ""

# Initialize extraction tracker
tracker = ExtractionTracker(input_plot, prompt_name)
tracker.initialize_stages(EXTRACT_STAGES)

# If article info was provided, add it as initial context
if article_info_json:
    stage_context["article_info"] = article_info_json

# -----------------------------------------------------------------------------
# Stage 1: extraction (and additional stages)
# Load the complete extraction schema
extraction_schema = load_extraction_schema()
complete_schema_str = json.dumps(extraction_schema, indent=2)

# Initialize accumulated facts with article info if provided
accumulated_facts = {}
if article_info_json:
    accumulated_facts = json.loads(article_info_json)

# Create an initial progress file immediately so the UI can show 0% from the start.
save_stage_update(
    version_dir,
    "STARTING",
    0,
    len(EXTRACT_STAGES),
    accumulated_facts,
    0,
    console_output="Extraction starting..."
)

# Stage processing loop
for stage_index, stage_name in enumerate(EXTRACT_STAGES):
    # Get the prompt text from the prompts module
    if not hasattr(prompts_module, stage_name):
        print(f"Error: Stage '{stage_name}' not found in prompts module")
        sys.exit(1)
    
    # Emit a "stage started" update where progress reflects only completed stages.
    save_stage_update(
        version_dir,
        stage_name,
        stage_index,
        len(EXTRACT_STAGES),
        accumulated_facts,
        0,
        console_output=console_timeline
    )

    tracker.start_stage(stage_name)
    stage_start_time = time.time()
    
    prompt_payload = getattr(prompts_module, stage_name)

    # Optional extra inputs for specific stages (kept opt-in via placeholders)
    csv_text_for_stage: Optional[str] = None
    csv_diagnostics_json_for_stage: Optional[str] = None
    stage6_output_json_for_stage: Optional[str] = None
    stage7_output_json_for_stage: Optional[str] = None

    if "{csv_text}" in prompt_payload or "{csv_diagnostics_json}" in prompt_payload:
        # Prefer the preserved CSV from the marker_facts-emitting stage
        candidate_csv = normalize_csv_to_wide(stage_context.get("_saved_csv", "") or "")
        if candidate_csv and not _csv_has_usable_curve_points(candidate_csv):
            candidate_csv = ""
        if not candidate_csv:
            # fallback: search backwards for nearest stage with usable CSV
            candidate_csv, candidate_stage = _find_best_prior_csv(stage_context, EXTRACT_STAGES, stage_index)
            if candidate_csv and candidate_stage:
                print(f"[CSV CONTEXT] Using prior stage CSV from {candidate_stage}")

        candidate_csv = normalize_csv_to_wide(candidate_csv or "")
        csv_text_for_stage = candidate_csv

        if "{csv_diagnostics_json}" in prompt_payload:
            # Compute deterministically from CSV text
            diagnostics_start = time.perf_counter()
            diagnostics_obj = compute_csv_diagnostics_from_text(csv_text_for_stage or "")
            diagnostics_ms = int((time.perf_counter() - diagnostics_start) * 1000)
            csv_diagnostics_json_for_stage = json.dumps(diagnostics_obj, indent=2)

            # Debug marker: make it unambiguous that Stage 6a ran.
            # This writes to the debug folder only; it does not alter stage inputs.
            if DEBUG_VERBOSE:
                try:
                    stage6a_debug = {
                        "stage6a_ran": True,
                        "requested_by_stage_index": stage_index,
                        "requested_by_stage_name": stage_name,
                        "compute_ms": diagnostics_ms,
                        "csv_text_chars": len(csv_text_for_stage or ""),
                        "csv_text_lines": (csv_text_for_stage or "").count("\n") + (1 if (csv_text_for_stage or "") else 0),
                        "csv_diagnostics_chars": len(csv_diagnostics_json_for_stage or ""),
                        "diagnostics": diagnostics_obj,
                    }
                    with open(
                        os.path.join(debug_dir, f"stage_06a_{stage_index+1:02d}_{stage_name}_csv_diagnostics.json"),
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(stage6a_debug, f, indent=2)

                    print(
                        f"[DEBUG] Stage 6a ran for {stage_name}: "
                        f"compute_ms={diagnostics_ms} csv_chars={len(csv_text_for_stage or '')}"
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to write Stage 6a marker for {stage_name}: {e}")

    if "{stage6_output_json}" in prompt_payload:
        stage6_output_json_for_stage = stage_context.get("EXTRACT_STAGE_6", "")
    if "{stage7_output_json}" in prompt_payload:
        stage7_output_json_for_stage = stage_context.get("EXTRACT_STAGE_7", "")
    
    # Format prompt with complete schema and accumulated facts
    accumulated_facts_str = json.dumps(accumulated_facts, indent=2) if accumulated_facts else "Empty (no facts extracted yet)"

    # Render known placeholders only (do not treat arbitrary '{...}' as formatting).
    # This prevents crashes when prompts include literal JSON examples.
    placeholder_values = {
        "complete_schema": complete_schema_str,
        "accumulated_facts": accumulated_facts_str,
        "replot_path": replot_plot,
        "csv_text": csv_text_for_stage or "",
        "csv_diagnostics_json": csv_diagnostics_json_for_stage or "",
        "stage6_output_json": stage6_output_json_for_stage or "",
        "stage7_output_json": stage7_output_json_for_stage or "",
    }
    for placeholder_key, placeholder_val in placeholder_values.items():
        token = "{" + placeholder_key + "}"
        if token in prompt_payload:
            prompt_payload = prompt_payload.replace(token, str(placeholder_val))

    # Debug logging: record rendered prompt and key input sizes before calling the model
    if DEBUG_VERBOSE:
        try:
            stage_debug = {
                "stage_index": stage_index,
                "stage_name": stage_name,
                "no_image_stage": stage_name in NO_IMAGE_STAGES,
                "prompt_chars": len(prompt_payload or ""),
                "accumulated_facts_chars": len(accumulated_facts_str or ""),
                "csv_text_chars": len(csv_text_for_stage or "") if csv_text_for_stage is not None else 0,
                "csv_diagnostics_chars": len(csv_diagnostics_json_for_stage or "") if csv_diagnostics_json_for_stage is not None else 0,
            }
            with open(
                os.path.join(debug_dir, f"stage_{stage_index+1:02d}_{stage_name}_inputs.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(stage_debug, f, indent=2)

            with open(
                os.path.join(debug_dir, f"stage_{stage_index+1:02d}_{stage_name}_prompt.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(prompt_payload or "")

            # Print a quick one-liner for live monitoring
            print(
                f"[DEBUG] Stage {stage_index+1}/{len(EXTRACT_STAGES)} {stage_name}: "
                f"no_image={stage_name in NO_IMAGE_STAGES} prompt_chars={len(prompt_payload or '')}"
            )
        except Exception as e:
            print(f"[DEBUG] Failed to write debug prompt for {stage_name}: {e}")
    elif "{data_context}" in prompt_payload:
        # Fallback for old-style prompts
        accumulated_context = ""
        if article_info_json:
            accumulated_context = f"=== Article Information ===\n{article_info_json}\n\n"
        
        for prev_stage_name in EXTRACT_STAGES[:stage_index]:
            if prev_stage_name in stage_context:
                accumulated_context += f"\n=== {prev_stage_name} ===\n{stage_context[prev_stage_name]}\n"
        
        prompt_payload = prompt_payload.format(
            data_context=accumulated_context.strip() if accumulated_context else "No previous context",
            replot_path=replot_plot,
        )

    if stage_name in NO_IMAGE_STAGES:
        messages = [{"role": "user", "content": prompt_payload}]
    else:
        messages = create_Q_1p([[base64_image, prompt_payload]])
    conversation_log.extend(messages)

    # Model call (add timing + persist raw response in debug mode)
    call_start = time.time()
    messages, result_text = prompt_llm(client, messages)
    call_ms = (time.time() - call_start) * 1000
    if DEBUG_VERBOSE:
        try:
            with open(
                os.path.join(debug_dir, f"stage_{stage_index+1:02d}_{stage_name}_response.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(result_text or "")

            with open(
                os.path.join(debug_dir, f"stage_{stage_index+1:02d}_{stage_name}_timing.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"model_call_ms": call_ms}, f, indent=2)
        except Exception as e:
            print(f"[DEBUG] Failed to write debug response for {stage_name}: {e}")
    conversation_log.append({"role": "assistant", "content": result_text})
    stage_context[stage_name] = result_text
    
    stage_time = (time.time() - stage_start_time) * 1000  # Convert to ms
    
    # Parse stage output as JSON (supports ```json fenced blocks) and merge into accumulated facts
    result_json = _extract_json_object_from_text(result_text)
    if isinstance(result_json, dict):
        for key in result_json:
            if (
                key in accumulated_facts
                and isinstance(accumulated_facts.get(key), dict)
                and isinstance(result_json.get(key), dict)
            ):
                accumulated_facts[key].update(result_json[key])
            else:
                accumulated_facts[key] = result_json[key]
    
    # Check for abort signal (supports legacy and structured formats)
    abort, abort_reason, abort_confidence = detect_abort_signal(result_text)

    # Extract facts from CSV if this is an extraction stage
    facts = None
    if "csv" in result_text.lower() or stage_index == len(EXTRACT_STAGES) - 1:
        facts = tracker.extract_facts_from_csv(result_text, stage_name)
    
    # Save CSV immediately after data extraction stage (stage that emits marker_facts)
    # This ensures CSV is preserved and can be reused by later stages
    if _result_has_marker_facts(result_json):
        try:
            csv_data = extract_csv_from_text(result_text)
            print(f"[DEBUG DATA EXTRACTION] Initial extracted CSV lines: {len(csv_data.split(chr(10)))}")
            print(f"[DEBUG DATA EXTRACTION] First line: {csv_data.split(chr(10))[0][:100] if csv_data else 'EMPTY'}")
            
            # Try to rebuild CSV from JSON curves for better quality
            # This avoids using the LLM's often-malformed csv_output string
            try:
                accumulated_facts_copy = accumulated_facts.copy()
                # Try to parse result_text as JSON to get fresh marker_facts
                result_json = _extract_json_object_from_text(result_text)
                
                if isinstance(result_json, dict) and "marker_facts" in result_json:
                    print(f"[DEBUG DATA EXTRACTION] Found marker_facts in result_json")
                    curves = result_json["marker_facts"].get("curves", [])
                    print(f"[DEBUG DATA EXTRACTION] Number of curves: {len(curves)}")
                    if curves:
                        print(f"[DEBUG DATA EXTRACTION] First curve: {curves[0].get('curve_label')} with {len(curves[0].get('points', []))} points")
                    
                    accumulated_facts_copy["marker_facts"] = result_json["marker_facts"]
                    rebuilt_csv = rebuild_csv_from_json_curves(accumulated_facts_copy)
                    if rebuilt_csv:
                        print(f"[DEBUG DATA EXTRACTION] Successfully rebuilt CSV from JSON curves")
                        print(f"[DEBUG DATA EXTRACTION] Rebuilt CSV first line: {rebuilt_csv.split(chr(10))[0][:100]}")
                        csv_data = rebuilt_csv
                    else:
                        print(f"[DEBUG DATA EXTRACTION] rebuild_csv_from_json_curves returned None")
                else:
                    print(f"[DEBUG DATA EXTRACTION] No marker_facts found in result_json or result_json is None")
                    if result_json:
                        print(f"[DEBUG DATA EXTRACTION] result_json keys: {result_json.keys()}")
            except Exception as e:
                print(f"[DEBUG DATA EXTRACTION] Could not rebuild CSV from JSON: {e}, using extracted CSV")
                import traceback
                traceback.print_exc()
            
            normalized_csv_data = normalize_csv_to_wide(csv_data or "")
            if normalized_csv_data and _csv_has_usable_curve_points(normalized_csv_data):
                # Store in a module-level variable for reuse if Stage 5 fails
                stage_context["_saved_csv"] = normalized_csv_data
                stage_context["_saved_csv_stage"] = stage_name
                csv_backup_path = output_out + f"_data_{stage_name.lower()}"
                with open(csv_backup_path, "w", encoding="utf-8") as f:
                    f.write(normalized_csv_data)
                print(f"CSV data successfully saved from {stage_name}")
            elif csv_data:
                print(f"[CSV REJECTED] Extracted CSV from {stage_name} was not usable; ignoring it.")
        except Exception as e:
            print(f"Warning: Could not save CSV from {stage_name}: {e}")
    
    # Use stage confidence if abort provided a confidence, else default
    stage_confidence = abort_confidence if abort_confidence is not None else 0.7
    tracker.complete_stage(stage_name, result_text, confidence=stage_confidence, 
                          execution_time_ms=stage_time, facts=facts)

    # REQUIRED: after each stage completes, dump full accumulated facts to console
    # as an immutable snapshot (timeline semantics; no overwriting).
    stage_dump = _format_stage_facts_dump(stage_index + 1, accumulated_facts)
    console_timeline += stage_dump
    _safe_console_print(stage_dump)
    
    # Save real-time progress update for web UI (completed stage count increments here)
    save_stage_update(
        version_dir,
        stage_name,
        stage_index + 1,
        len(EXTRACT_STAGES),
        accumulated_facts,
        stage_time,
        console_output=console_timeline,
    )

    # Optional pacing between stages (helps avoid rate-limit bursts).
    if BACKOFF_STAGE_SLEEP_S > 0 and stage_index < (len(EXTRACT_STAGES) - 1):
        time.sleep(BACKOFF_STAGE_SLEEP_S)

    # Early stop on abort
    if abort:
        reason_display = abort_reason or "Abort signaled by prompt"
        tracker.fail_stage(stage_name, f"ABORT: {reason_display}")

        # Write legacy data file for compatibility
        with open(output_out + "_data", "w", encoding="utf-8") as file:
            file.write("None")

        # Write abort meta information
        abort_meta = {
            "stage": stage_name,
            "stage_index": stage_index,
            "reason": reason_display,
            "confidence": abort_confidence,
            "accumulated_facts": accumulated_facts,
        }
        try:
            with open(output_out + "_abort", "w", encoding="utf-8") as f:
                json.dump(abort_meta, f, indent=2)
        except Exception:
            pass

        print(f"ABORTED at stage '{stage_name}' - {reason_display}")
        tracker.save_tracking_report(output_out + "_tracking")
        print(f"VERSION_DIR:{version_dir}")
        sys.exit(0)

# Mark extraction as complete
tracker.mark_complete()

# -----------------------------------------------------------------------------
# Stage outputs
# -----------------------------------------------------------------------------
# Extract CSV from final stage
final_stage = EXTRACT_STAGES[-1]
data_raw = stage_context.get(final_stage, "")
data_from_final = extract_csv_from_text(data_raw)
final_csv_valid = _csv_has_usable_curve_points(data_from_final)

# Check if we have a saved CSV from the data extraction stage
stage4_csv = stage_context.get("_saved_csv", "")
saved_csv_stage = stage_context.get("_saved_csv_stage", "")
saved_csv_valid = _csv_has_usable_curve_points(stage4_csv)

# Prefer final-stage CSV when the final stage differs from the data-extraction stage.
if data_from_final and final_csv_valid and (not saved_csv_stage or final_stage != saved_csv_stage):
    data = data_from_final
    print(f"Using CSV from final stage: {final_stage}")
elif stage4_csv and saved_csv_valid:
    data = stage4_csv
    stage_label = saved_csv_stage or "data extraction stage"
    print(f"Using CSV from {stage_label} (data extraction stage)")
elif stage4_csv:
    data = stage4_csv
    stage_label = saved_csv_stage or "data extraction stage"
    print(f"Using CSV from {stage_label} (fallback; validation unavailable)")
elif data_from_final:
    data = data_from_final
    print(f"Using CSV from final stage (fallback): {final_stage}")
else:
    data = data_from_final
    print("Warning: No saved CSV found, using final stage output")

# Normalize CSV to wide format if it came back in long format
data = normalize_csv_to_wide(data)

# If selected CSV is unusable, try deterministic rebuild from accumulated marker_facts
if data and not _csv_has_usable_curve_points(data):
    rebuilt_data = rebuild_csv_from_json_curves(accumulated_facts)
    if rebuilt_data and _csv_has_usable_curve_points(rebuilt_data):
        print("Recovered CSV by rebuilding from accumulated marker_facts curves")
        data = rebuilt_data
    else:
        print("Warning: Selected CSV content is not usable curve data")
        data = ""

# If still no valid CSV, try to recover from earlier stages
if not data or data.startswith("Here is the") or data == "accumulated_facts":
    print("Warning: Could not extract valid CSV data from any stage")
    data = ""

# For backward compatibility, also check if there's a CODE_PLOT stage result
code = ""
if hasattr(prompts_module, "CODE_PLOT"):
    # Generate matplotlib code to replot the data
    code_template = prompts_module.CODE_PLOT
    
    # Check which format variables the template uses
    if "{accumulated_facts}" in code_template:
        # New format: use accumulated facts as JSON
        accumulated_facts_str = json.dumps(accumulated_facts, indent=2)
        code_prompt = code_template.format(
            accumulated_facts=accumulated_facts_str,
            replot_path=replot_plot
        )
    else:
        # Old format: use data_context (CSV only)
        code_prompt = code_template.format(
            data_context=data,
            replot_path=replot_plot
        )
    
    messages = create_Q_1p([[base64_image, code_prompt]])
    conversation_log.extend(messages)
    messages, code = prompt_llm(client, messages)
    code = clean_code_response(code)
    conversation_log.append({"role": "assistant", "content": code})

with open(output_out + "_data", "w", encoding="utf-8") as file:
    file.write(data)

# Save a clean CSV file with user-friendly naming
if data and data != "None" and not data.startswith("Here is the"):
    clean_csv_path = os.path.join(version_dir, f"{base_name}_extracted.csv")
    with open(clean_csv_path, "w", encoding="utf-8") as file:
        file.write(data)
    print(f"CSV saved to: {clean_csv_path}")

# -----------------------------------------------------------------------------
# Code execution and repair loop
# -----------------------------------------------------------------------------
error_output = None
if code:
    print("Replotting with extracted data... ", end="", flush=True)
    import builtins

    _orig_import = builtins.__import__
    _patched_once = {"done": False}

    def _install_savefig_patch_if_possible():
        if _patched_once["done"]:
            return
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import matplotlib.figure as mpl_figure  # type: ignore
        except Exception:
            return

        # This code runs inside a long-lived process that may execute many plots.
        # If we wrap savefig more than once, the "original" can become our own wrapper,
        # which then calls itself and triggers RecursionError.
        def _is_already_patched(func):
            return bool(getattr(func, "_plotextract_is_patched", False))

        def _get_original(func):
            return getattr(func, "_plotextract_original", func)

        if _is_already_patched(plt.savefig) and _is_already_patched(mpl_figure.Figure.savefig):
            _patched_once["done"] = True
            return

        _orig_plt_savefig = _get_original(plt.savefig)
        _orig_fig_savefig = _get_original(mpl_figure.Figure.savefig)

        _in_overlay_save = {"active": False}

        def _derive_overlay_paths_from_replot(path: str):
            p = str(path)
            if '-replot.' in p:
                return (
                    p.replace('-replot.', '-replot_overlay_full.', 1),
                    p.replace('-replot.', '-replot_overlay_minmax.', 1),
                )
            if p.lower().endswith('.png'):
                stem = p[:-4]
                return stem + '_overlay_full.png', stem + '_overlay_minmax.png'
            return p + '_overlay_full', p + '_overlay_minmax'

        def _derive_overlay_axes_paths_from_replot(path: str):
            p = str(path)
            if '-replot.' in p:
                return (
                    p.replace('-replot.', '-replot_overlay_axes_full.', 1),
                    p.replace('-replot.', '-replot_overlay_axes_minmax.', 1),
                )
            if p.lower().endswith('.png'):
                stem = p[:-4]
                return stem + '_overlay_axes_full.png', stem + '_overlay_axes_minmax.png'
            return p + '_overlay_axes_full', p + '_overlay_axes_minmax'

        def _derive_overlay_curve_path_from_replot(path: str, idx: int, axis_mode: str):
            p = str(path)
            tag = f"-replot_overlay_curve_{idx}_{axis_mode}."
            if '-replot.' in p:
                return p.replace('-replot.', tag, 1)
            suffix = f"_overlay_curve_{idx}_{axis_mode}"
            if p.lower().endswith('.png'):
                return p[:-4] + suffix + '.png'
            return p + suffix

        def _format_tick_val(v):
            try:
                return f"{float(v):.6g}"
            except Exception:
                try:
                    return str(v)
                except Exception:
                    return ''

        def _save_transparent_replot_overlay(fig, out_path: str, axis_mode: str = 'full', save_kwargs=None):
            import os
            try:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
            except Exception:
                pass

            fig_patch_alpha = None
            fig_legend_state = []
            try:
                fig_patch_alpha = fig.patch.get_alpha()
                fig.patch.set_alpha(0.0)
            except Exception:
                fig_patch_alpha = None

            try:
                for lg in list(getattr(fig, 'legends', []) or []):
                    try:
                        vis = bool(lg.get_visible())
                    except Exception:
                        vis = True
                    fig_legend_state.append((lg, vis))
                    try:
                        lg.set_visible(False)
                    except Exception:
                        pass
            except Exception:
                fig_legend_state = []

            per_ax_state = []
            try:
                axes = getattr(fig, 'axes', []) or []
            except Exception:
                axes = []

            for ax in axes:
                st = {
                    'ax': ax,
                    'patch_alpha': None,
                    'lines': [],
                    'collections': [],
                    'legend': None,
                    'legend_visible': None,
                    'x_major_locator': None,
                    'x_major_formatter': None,
                    'x_minor_locator': None,
                    'x_minor_formatter': None,
                    'y_major_locator': None,
                    'y_major_formatter': None,
                    'y_minor_locator': None,
                    'y_minor_formatter': None,
                }
                try:
                    st['patch_alpha'] = ax.patch.get_alpha()
                    ax.patch.set_alpha(0.0)
                except Exception:
                    st['patch_alpha'] = None

                try:
                    for ln in ax.get_lines() or []:
                        try:
                            st['lines'].append((ln, ln.get_color(), ln.get_alpha(), ln.get_linewidth()))
                            ln.set_color('#ef4444')
                            if ln.get_alpha() is None:
                                ln.set_alpha(1.0)
                        except Exception:
                            continue
                except Exception:
                    pass

                try:
                    lg = ax.get_legend()
                    if lg is not None:
                        st['legend'] = lg
                        try:
                            st['legend_visible'] = bool(lg.get_visible())
                        except Exception:
                            st['legend_visible'] = True
                        try:
                            lg.set_visible(False)
                        except Exception:
                            pass
                except Exception:
                    pass

                try:
                    for coll in getattr(ax, 'collections', []) or []:
                        try:
                            ec = coll.get_edgecolor()
                        except Exception:
                            ec = None
                        try:
                            fc = coll.get_facecolor()
                        except Exception:
                            fc = None
                        st['collections'].append((coll, ec, fc))
                        try:
                            coll.set_edgecolor('#ef4444')
                        except Exception:
                            pass
                        try:
                            coll.set_facecolor('#ef4444')
                        except Exception:
                            pass
                except Exception:
                    pass

                if axis_mode == 'minmax':
                    try:
                        st['x_major_locator'] = ax.xaxis.get_major_locator()
                        st['x_major_formatter'] = ax.xaxis.get_major_formatter()
                        st['x_minor_locator'] = ax.xaxis.get_minor_locator()
                        st['x_minor_formatter'] = ax.xaxis.get_minor_formatter()
                        st['y_major_locator'] = ax.yaxis.get_major_locator()
                        st['y_major_formatter'] = ax.yaxis.get_major_formatter()
                        st['y_minor_locator'] = ax.yaxis.get_minor_locator()
                        st['y_minor_formatter'] = ax.yaxis.get_minor_formatter()

                        xmin, xmax = ax.get_xlim()
                        ymin, ymax = ax.get_ylim()
                        ax.set_xticks([xmin, xmax])
                        ax.set_yticks([ymin, ymax])
                        ax.set_xticklabels([_format_tick_val(xmin), _format_tick_val(xmax)])
                        ax.set_yticklabels([_format_tick_val(ymin), _format_tick_val(ymax)])
                        try:
                            ax.minorticks_off()
                        except Exception:
                            pass
                    except Exception:
                        pass

                per_ax_state.append(st)

            dpi = None
            bbox_inches = None
            pad_inches = None
            try:
                sk = save_kwargs or {}
                dpi = sk.get('dpi', None)
                bbox_inches = sk.get('bbox_inches', None)
                pad_inches = sk.get('pad_inches', None)
            except Exception:
                dpi = None
                bbox_inches = None
                pad_inches = None

            if dpi is None:
                dpi = 300
            if bbox_inches is None:
                bbox_inches = 'tight'

            try:
                _orig_fig_savefig(
                    fig,
                    out_path,
                    dpi=dpi,
                    bbox_inches=bbox_inches,
                    pad_inches=pad_inches,
                    transparent=True,
                    facecolor='none',
                    edgecolor='none',
                )
            except Exception:
                pass
            finally:
                for st in per_ax_state:
                    ax = st.get('ax')
                    if not ax:
                        continue
                    try:
                        if st.get('patch_alpha') is not None:
                            ax.patch.set_alpha(st['patch_alpha'])
                    except Exception:
                        pass

                    for (ln, c, a, lw) in st.get('lines', []) or []:
                        try:
                            ln.set_color(c)
                            ln.set_alpha(a)
                            ln.set_linewidth(lw)
                        except Exception:
                            continue

                    for (coll, ec, fc) in st.get('collections', []) or []:
                        try:
                            if ec is not None:
                                coll.set_edgecolor(ec)
                        except Exception:
                            pass
                        try:
                            if fc is not None:
                                coll.set_facecolor(fc)
                        except Exception:
                            pass

                    try:
                        lg = st.get('legend')
                        if lg is not None:
                            lg.set_visible(bool(st.get('legend_visible')))
                    except Exception:
                        pass

                    if axis_mode == 'minmax':
                        try:
                            if st.get('x_major_locator') is not None:
                                ax.xaxis.set_major_locator(st['x_major_locator'])
                            if st.get('x_major_formatter') is not None:
                                ax.xaxis.set_major_formatter(st['x_major_formatter'])
                            if st.get('x_minor_locator') is not None:
                                ax.xaxis.set_minor_locator(st['x_minor_locator'])
                            if st.get('x_minor_formatter') is not None:
                                ax.xaxis.set_minor_formatter(st['x_minor_formatter'])
                            if st.get('y_major_locator') is not None:
                                ax.yaxis.set_major_locator(st['y_major_locator'])
                            if st.get('y_major_formatter') is not None:
                                ax.yaxis.set_major_formatter(st['y_major_formatter'])
                            if st.get('y_minor_locator') is not None:
                                ax.yaxis.set_minor_locator(st['y_minor_locator'])
                            if st.get('y_minor_formatter') is not None:
                                ax.yaxis.set_minor_formatter(st['y_minor_formatter'])
                        except Exception:
                            pass

                try:
                    if fig_patch_alpha is not None:
                        fig.patch.set_alpha(fig_patch_alpha)
                except Exception:
                    pass

                for (lg, was_vis) in fig_legend_state:
                    try:
                        lg.set_visible(bool(was_vis))
                    except Exception:
                        pass

        def _save_axes_and_curve_overlay_layers(fig, replot_path: str, save_kwargs=None):
            try:
                axes = getattr(fig, 'axes', []) or []
            except Exception:
                axes = []

            lines = []
            for ax in axes:
                try:
                    for ln in ax.get_lines() or []:
                        lines.append(ln)
                except Exception:
                    continue

            if not lines:
                return

            vis_state = []
            for ln in lines:
                try:
                    vis_state.append(bool(ln.get_visible()))
                except Exception:
                    vis_state.append(True)

            axes_full, axes_minmax = _derive_overlay_axes_paths_from_replot(replot_path)
            try:
                for ln in lines:
                    try:
                        ln.set_visible(False)
                    except Exception:
                        pass

                _save_transparent_replot_overlay(fig, axes_full, axis_mode='full', save_kwargs=save_kwargs)
                _save_transparent_replot_overlay(fig, axes_minmax, axis_mode='minmax', save_kwargs=save_kwargs)

                for i, ln in enumerate(lines, start=1):
                    for other in lines:
                        try:
                            other.set_visible(other is ln)
                        except Exception:
                            pass
                    out_full = _derive_overlay_curve_path_from_replot(replot_path, i, 'full')
                    out_minmax = _derive_overlay_curve_path_from_replot(replot_path, i, 'minmax')
                    _save_transparent_replot_overlay(fig, out_full, axis_mode='full', save_kwargs=save_kwargs)
                    _save_transparent_replot_overlay(fig, out_minmax, axis_mode='minmax', save_kwargs=save_kwargs)
            finally:
                for ln, was_vis in zip(lines, vis_state):
                    try:
                        ln.set_visible(bool(was_vis))
                    except Exception:
                        pass

        def _maybe_emit_replot_overlay_variants(fig, save_args, save_kwargs):
            import os
            if _in_overlay_save.get('active'):
                return

            out_path = None
            try:
                if save_args and isinstance(save_args[0], (str, os.PathLike)):
                    out_path = os.fspath(save_args[0])
            except Exception:
                out_path = None

            if not out_path:
                return
            try:
                if not str(out_path).lower().endswith('.png'):
                    return
            except Exception:
                return

            try:
                if os.path.abspath(str(out_path)) != os.path.abspath(str(replot_plot)):
                    return
            except Exception:
                if str(out_path) != str(replot_plot):
                    return

            overlay_full, overlay_minmax = _derive_overlay_paths_from_replot(str(out_path))
            try:
                _in_overlay_save['active'] = True
                _save_transparent_replot_overlay(fig, overlay_full, axis_mode='full', save_kwargs=save_kwargs)
                _save_transparent_replot_overlay(fig, overlay_minmax, axis_mode='minmax', save_kwargs=save_kwargs)
                _save_axes_and_curve_overlay_layers(fig, str(out_path), save_kwargs=save_kwargs)
            finally:
                _in_overlay_save['active'] = False

        def _apply_replot_axis_policy(fig):
            try:
                axes = getattr(fig, 'axes', []) or []
                for ax in axes:
                    try:
                        # Y: always start at 0.001 (works for log plots too)
                        if ax.get_yscale() == 'log':
                            _, cur_top = ax.get_ylim()
                            top = cur_top if (cur_top is not None and cur_top > 0.001) else 0.01
                            ax.set_ylim(0.001, top)
                        else:
                            ax.set_ylim(bottom=0.001)

                        # X: start at 0 only for linear axes
                        if ax.get_xscale() == 'linear':
                            ax.set_xlim(left=0.0)
                    except Exception:
                        continue
            except Exception:
                pass

        def _patched_plt_savefig(*args, **kwargs):
            try:
                _apply_replot_axis_policy(plt.gcf())
            except Exception:
                pass
            ret = _orig_plt_savefig(*args, **kwargs)
            try:
                _maybe_emit_replot_overlay_variants(plt.gcf(), args, kwargs)
            except Exception:
                pass
            return ret

        def _patched_fig_savefig(self, *args, **kwargs):
            _apply_replot_axis_policy(self)
            ret = _orig_fig_savefig(self, *args, **kwargs)
            try:
                _maybe_emit_replot_overlay_variants(self, args, kwargs)
            except Exception:
                pass
            return ret

        _patched_plt_savefig._plotextract_is_patched = True
        _patched_plt_savefig._plotextract_original = _orig_plt_savefig
        _patched_fig_savefig._plotextract_is_patched = True
        _patched_fig_savefig._plotextract_original = _orig_fig_savefig

        plt.savefig = _patched_plt_savefig
        mpl_figure.Figure.savefig = _patched_fig_savefig
        _patched_once["done"] = True

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        # Patch when matplotlib pyplot is imported in common patterns.
        if name == 'matplotlib.pyplot' or (name == 'matplotlib' and fromlist and 'pyplot' in fromlist):
            _install_savefig_patch_if_possible()
        return mod

    builtins.__import__ = _patched_import
    try:
        # In case pyplot is already imported in this process.
        _install_savefig_patch_if_possible()
        exec(code)
        print("FINISHED")
    except Exception:
        error_output = traceback.format_exc()
    finally:
        builtins.__import__ = _orig_import
else:
    print("No CODE_PLOT prompt found, skipping replot generation")
    error_output = "SKIP_REPLOT"

# If code executed but didn't write the expected file, force a repair pass.
if code and not error_output and not os.path.exists(replot_plot):
    error_output = (
        "Replot output file was not created at the required path.\n"
        f"Expected replot_path: {replot_plot}\n"
        "Your code MUST save the image to that exact path via plt.savefig(replot_path, ...).\n"
        "Do not invent output directories or alternate filenames."
    )

if error_output and error_output != "SKIP_REPLOT":
    print("ERROR in replot code, fixing error...")
    if hasattr(prompts_module, "CODE_FIX"):
        fix_prompt = prompts_module.CODE_FIX
    else:
        fix_prompt = "Fix the code above. Respond with corrected code only."

    # Always remind the model of the required save path.
    fix_prompt = (
        fix_prompt
        + f"\n\nCRITICAL: The corrected code MUST save the plot to EXACTLY this path: {replot_plot}\n"
        + "Use os.makedirs(os.path.dirname(replot_path), exist_ok=True) and plt.savefig(replot_path, ...)."
    )
    repair_messages = conversation_log + [{"role": "user", "content": error_output + fix_prompt}]
    _, code = prompt_llm(client, repair_messages)
    code = clean_code_response(code)
    try:
        exec(code)
        print("SUCCESS, error fixed")
        error_output = None
    except Exception:
        error_output = traceback.format_exc()

if error_output and error_output != "SKIP_REPLOT":
    print("FAILED - need to redo {input_plot}")
    print(error_output)

if code:
    with open(output_out + "_code", "w", encoding="utf-8") as file:
        file.write(code)
    with open(output_out + "_conversation", "a", encoding="utf-8") as file:
        json.dump(conversation_log + [{"role": "assistant", "content": code.replace("\n", "\\n")}], file)

# -----------------------------------------------------------------------------
# Validation (uses prompts from the loaded prompt set)
# -----------------------------------------------------------------------------

# Only validate if we successfully generated a replot
if code and error_output != "SKIP_REPLOT" and not error_output:
    comparison_original = api_image_path if original_ext == ".svg" else input_plot
    stacked = stack_images_vertically(comparison_original, replot_plot, "yes", version_dir, prompt_name, version_num, key_tag=key_tag, output_tag=(output_tag or None))
else:
    stacked = None
    print("Skipping validation (no replot generated)")

if stacked:
    print("Comparing source and replot... ", end="", flush=True)
    wrong = False
    wrong_why = ""
    validation_details = {}

    def run_validation(prompt_text):
        # The stacked comparison image is always saved as PNG.
        # Do not rely on PNGJPG (derived from the original input extension) or the API can reject the request.
        stacked_b64 = encode_image(stacked)
        msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{stacked_b64}"}},
            ],
        }]
        _, validate_resp = prompt_llm(client, msg)
        return validate_resp

    # Get comparison prompts from the loaded prompts module
    compare_x = getattr(prompts_module, 'COMPARE_X', 'Do the X axes match?')
    compare_y = getattr(prompts_module, 'COMPARE_Y', 'Do the Y axes match?')
    compare_number = getattr(prompts_module, 'COMPARE_NUMBER', 'Do the number of points match?')
    compare_trend = getattr(prompts_module, 'COMPARE_TREND', 'Does the trend match?')

    validate_x = run_validation(compare_x)
    validation_details['x_axis'] = validate_x
    print(f"\n\nAxis x (result: {validate_x})")
    if "no" in validate_x.lower().strip()[:10]:
        wrong = True
        wrong_why += "X; "

    validate_y = run_validation(compare_y)
    validation_details['y_axis'] = validate_y
    print(f"Axis y (result: {validate_y})")
    if "no" in validate_y.lower().strip()[:10]:
        wrong = True
        wrong_why += "Y; "

    validate_n = run_validation(compare_number)
    validation_details['num_points'] = validate_n
    print(f"Points n (result: {validate_n})")
    if "no" in validate_n.lower().strip()[:10]:
        wrong = True
        wrong_why += "N; "

    validate_t = run_validation(compare_trend)
    validation_details['trend'] = validate_t
    print(f"Trends (result: {validate_t})")
    if "no" in validate_t.lower().strip()[:10]:
        wrong = True
        wrong_why += "T"

    with open(output_out + "_validate", "w", encoding="utf-8") as file:
        if wrong:
            file.write("no")
        else:
            file.write("yes")
    if wrong:
        with open(output_out + "_validate_why", "w", encoding="utf-8") as file:
            file.write(wrong_why)
    result_flag = "no" if wrong else "yes"
    
    # Track validation results
    tracker.set_validation_result(result_flag, validation_details)
    print(f"\nFINISHED (result: {result_flag})")

    print("Stacking original and replotted images for comparison... ", end="", flush=True)
    stack_images_vertically(comparison_original, replot_plot, "no" if wrong else "yes", version_dir, prompt_name, version_num, key_tag=key_tag, output_tag=(output_tag or None))
    print("FINISHED")
else:
    print("Skipping visual validation (comparison image could not be generated)")
    with open(output_out + "_validate", "w", encoding="utf-8") as file:
        file.write("skipped")
    tracker.set_validation_result("skipped")

if original_ext == ".svg" and api_image_path != input_plot:
    try:
        os.unlink(api_image_path)
    except Exception:
        pass

# Save tracking report
tracker_report_path = output_out + "_tracking"
tracker.save_tracking_report(tracker_report_path)
tracker.print_summary()

# Save final completion status to progress file
save_stage_update(version_dir, "COMPLETE", len(EXTRACT_STAGES), len(EXTRACT_STAGES), 
                 accumulated_facts, 0, console_output="Extraction completed successfully")

print(f"VERSION_DIR:{version_dir}")
print("\n\n")
