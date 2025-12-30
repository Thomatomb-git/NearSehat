from __future__ import annotations

# details.py (LLM call - single call only)
# Output harus "recommend-ready": urgency, required/preferred facilities tokens, max distance, rationale, safety note.
# IMPORTANT: rationale dan safety_note harus berbahasa Inggris.

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import os
import re
from typing import Dict, List, Tuple, Any

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")  # default handled below

COMPLAINT_INPUT = 2


def _norm_token(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _system_prompt() -> str:
    return (
        "You are NearSehat's clinical triage and recommendation-criteria extractor.\n"
        "You must output STRICT JSON (no markdown, no extra keys) with the following keys:\n"
        "- urgency: one of 'low', 'medium', 'high'\n"
        "- required_facilities: array of facility tokens\n"
        "- preferred_facilities: array of facility tokens\n"
        "- max_distance_km: number\n"
        "- rationale: one short sentence in ENGLISH explaining the decision\n"
        "- safety_note: one short sentence in ENGLISH. If urgency is high, emphasize going to the nearest ER / emergency services.\n\n"
        "Rules:\n"
        "1) Use ONLY facility tokens from the provided Allowed Facility Tokens list.\n"
        "2) required_facilities must be minimal but necessary (avoid being too strict).\n"
        "3) preferred_facilities are nice-to-have.\n"
        "4) If symptoms suggest emergency, set urgency='high' and keep max_distance_km relatively small.\n"
        "5) If unclear, be conservative: urgency='medium', max_distance_km around 12.\n"
        "6) Never invent hospitals or facts.\n"
        "7) Output MUST be English for rationale and safety_note."
    )


def _user_prompt(complaint: str, allowed_tokens: List[str]) -> str:
    allowed_str = "\n".join(allowed_tokens)
    return (
        "Patient complaint / needs:\n"
        f"{complaint}\n\n"
        "Allowed Facility Tokens (choose only from this list):\n"
        f"{allowed_str}\n\n"
        "Output STRICT JSON with the exact keys described."
    )


def _sanitize_criteria(raw: Dict[str, Any], allowed_tokens: List[str]) -> Dict[str, Any]:
    allowed_set = {_norm_token(t) for t in allowed_tokens}

    urgency = str(raw.get("urgency", "medium")).strip().lower()
    if urgency not in {"low", "medium", "high"}:
        urgency = "medium"

    def _to_list(v) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return []

    req = [_norm_token(x) for x in _to_list(raw.get("required_facilities"))]
    pref = [_norm_token(x) for x in _to_list(raw.get("preferred_facilities"))]

    req = [t for t in req if t in allowed_set]
    pref = [t for t in pref if t in allowed_set]

    req = _dedupe_keep_order(req)
    pref = [t for t in _dedupe_keep_order(pref) if t not in set(req)]

    maxd_raw = raw.get("max_distance_km", 12)
    try:
        maxd = float(maxd_raw)
    except Exception:
        maxd = 12.0
    if maxd <= 0:
        maxd = 12.0
    if maxd > 100:
        maxd = 100.0

    rationale = str(raw.get("rationale", "")).strip() or "Criteria were extracted from your symptoms."
    safety = str(raw.get("safety_note", "")).strip() or (
        "If you have severe symptoms or feel unsafe, go to the nearest emergency room immediately."
        if urgency == "high"
        else "If symptoms worsen, seek medical help promptly."
    )

    # keep short
    if len(rationale) > 160:
        rationale = rationale[:157].rstrip() + "..."
    if len(safety) > 160:
        safety = safety[:157].rstrip() + "..."

    return {
        "urgency": urgency,
        "required_facilities": req,
        "preferred_facilities": pref,
        "max_distance_km": maxd,
        "rationale": rationale,
        "safety_note": safety,
    }


async def generate_recommendation_criteria(
    complaint: str,
    allowed_tokens: List[str],
) -> Tuple[bool, Dict[str, Any] | str]:
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is not set."

    if not allowed_tokens:
        return False, "Allowed facility tokens list is empty (data3.csv may be invalid)."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        def _req():
            return client.responses.create(
                model=OPENAI_MODEL or "gpt-5-mini",
                input=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _user_prompt(complaint, allowed_tokens)},
                ],
                text={"format": {"type": "json_object"}},
            )

        rsp = await asyncio.get_running_loop().run_in_executor(None, _req)

        payload = getattr(rsp, "output_text", None)
        if not payload:
            try:
                payload = rsp.output[0].content[0].text  # type: ignore
            except Exception:
                payload = None

        if not payload:
            return False, "Empty model response."

        raw = json.loads(payload)
        if not isinstance(raw, dict):
            return False, "Model response is not a JSON object."

        return True, _sanitize_criteria(raw, allowed_tokens)

    except json.JSONDecodeError:
        return False, "Model returned invalid JSON."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
