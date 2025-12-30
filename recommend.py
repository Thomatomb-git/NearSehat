from __future__ import annotations

# recommend.py (deterministic recommender + compact UI payload)
# Tidak ada LLM di sini. Semua kriteria datang dari details.py (single call).
# Output ke user harus dalam bahasa Inggris.

import csv
import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any


@dataclass(frozen=True)
class Hospital:
    kode: str
    nama: str
    jenis: str
    kelas: str
    alamat: str


@dataclass(frozen=True)
class FacilityInfo:
    facilities: Set[str]
    beds: Optional[int] = None


DEFAULT_DATA1_NAME = "data1.csv"
DEFAULT_DATA2_NAME = "data2.csv"
DEFAULT_DATA3_NAME = "data3.csv"


def _resolve_path(env_var: str, filename: str) -> Path:
    p = os.getenv(env_var)
    if p:
        return Path(p).expanduser()

    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        return (Path(data_dir).expanduser() / filename)

    candidates = [
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
    ]

    here = Path(__file__).resolve().parent
    candidates.extend([
        here / filename,
        here / "data" / filename,
    ])

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


def _norm_token(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _pretty_token(token: str) -> str:
    """
    Convert token like EMERGENCY_ROOM_24_JAM -> Emergency Room 24 Hours
    (Khusus 'Jam' kita translate jadi 'Hours' karena output harus English.)
    """
    t = (token or "").strip().replace("_", " ").lower()
    words = []
    for w in t.split():
        if not w:
            continue
        if w == "jam":
            words.append("Hours")
        elif w.isdigit():
            words.append(w)
        elif len(w) <= 3 and w.isalpha():
            words.append(w.upper())
        else:
            words.append(w.capitalize())
    return " ".join(words)


def _format_token_list(tokens: List[str]) -> str:
    if not tokens:
        return "None"
    return ", ".join(_pretty_token(t) for t in tokens)


def _is_truthy(v: str) -> bool:
    s = (v or "").strip().lower()
    return s in {"1", "true", "yes", "y", "ya", "ada", "available", "avail", "t"}


def _parse_int_safe(v: str) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return int(float(s))
    except Exception:
        return None


def _looks_like_beds_column(col: str) -> bool:
    c = (col or "").strip().lower()
    return ("bed" in c and "total" in c) or c in {"beds", "totalbeds", "total_beds", "total bed", "total beds"}


_LOADED = False
_HOSPITALS: List[Hospital] = []
_DIST: Dict[Tuple[str, str], float] = {}
_FAC: Dict[str, FacilityInfo] = {}
_ALLOWED_TOKENS: List[str] = []


def _ensure_loaded() -> None:
    """
    IMPORTANT BUGFIX:
    data3.csv header sering punya trailing space (contoh: 'Emergency room (24 jam) ').
    Kalau header kita .strip() lalu dipakai untuk row.get(...), key-nya tidak match -> facility kebaca kosong.
    Solusi: akses row.get() pakai header original, token tetap dibuat dari versi stripped.
    """
    global _LOADED, _HOSPITALS, _DIST, _FAC, _ALLOWED_TOKENS
    if _LOADED:
        return

    # -------------------------
    # data1.csv
    # -------------------------
    p1 = _resolve_path("DATA1_PATH", DEFAULT_DATA1_NAME)
    if not p1.exists():
        raise FileNotFoundError(f"data1.csv not found. Tried: {p1}. Set DATA1_PATH or DATA_DIR.")

    hospitals: List[Hospital] = []
    with p1.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("data1.csv has no header row.")

        fieldnames_stripped = {h.strip() for h in reader.fieldnames}
        required_cols = {"kode", "nama", "jenis", "kelas", "alamat"}
        missing = required_cols - fieldnames_stripped
        if missing:
            raise ValueError(f"data1.csv missing columns: {sorted(missing)}")

        for row in reader:
            kode = str(row.get("kode", "")).strip()
            if not kode:
                continue
            hospitals.append(
                Hospital(
                    kode=kode,
                    nama=str(row.get("nama", "")).strip(),
                    jenis=str(row.get("jenis", "")).strip(),
                    kelas=str(row.get("kelas", "")).strip(),
                    alamat=str(row.get("alamat", "")).strip(),
                )
            )

    # -------------------------
    # data2.csv
    # -------------------------
    p2 = _resolve_path("DATA2_PATH", DEFAULT_DATA2_NAME)
    if not p2.exists():
        raise FileNotFoundError(f"data2.csv not found. Tried: {p2}. Set DATA2_PATH or DATA_DIR.")

    dist: Dict[Tuple[str, str], float] = {}
    with p2.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("data2.csv has no header row.")
        fieldnames_stripped = {h.strip() for h in reader.fieldnames}
        required_cols = {"origin", "target", "distance"}
        missing = required_cols - fieldnames_stripped
        if missing:
            raise ValueError(f"data2.csv missing columns: {sorted(missing)}")

        for row in reader:
            o = str(row.get("origin", "")).strip()
            t = str(row.get("target", "")).strip()
            d = str(row.get("distance", "")).strip()
            if not o or not t or not d:
                continue
            try:
                dist[(o, t)] = float(d)
            except Exception:
                continue

    # -------------------------
    # data3.csv
    # -------------------------
    p3 = _resolve_path("DATA3_PATH", DEFAULT_DATA3_NAME)
    if not p3.exists():
        raise FileNotFoundError(f"data3.csv not found. Tried: {p3}. Set DATA3_PATH or DATA_DIR.")

    fac: Dict[str, FacilityInfo] = {}
    allowed_tokens_set: Set[str] = set()

    with p3.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("data3.csv has no header row.")

        original_fieldnames = list(reader.fieldnames)
        pairs = [(orig, orig.strip()) for orig in original_fieldnames]

        def find_col(canonical: str) -> str:
            for orig, stripped in pairs:
                if stripped.lower() == canonical.lower():
                    return orig
            raise ValueError(f"data3.csv missing required column: '{canonical}'")

        kode_col = find_col("kode")

        beds_col_orig: Optional[str] = None
        for orig, stripped in pairs:
            if _looks_like_beds_column(stripped):
                beds_col_orig = orig
                break

        facility_cols_orig: List[str] = []
        facility_tokens: List[str] = []

        for orig, stripped in pairs:
            if stripped.lower() in {"kode", "nama"}:
                continue
            if beds_col_orig and orig == beds_col_orig:
                continue

            facility_cols_orig.append(orig)
            tok = _norm_token(stripped)
            facility_tokens.append(tok)
            allowed_tokens_set.add(tok)

        for row in reader:
            kode = str(row.get(kode_col, "")).strip()
            if not kode:
                continue

            beds = None
            if beds_col_orig:
                beds = _parse_int_safe(row.get(beds_col_orig, ""))

            s: Set[str] = set()
            for orig, tok in zip(facility_cols_orig, facility_tokens):
                if _is_truthy(str(row.get(orig, ""))):
                    s.add(tok)

            fac[kode] = FacilityInfo(facilities=s, beds=beds)

    _HOSPITALS = hospitals
    _DIST = dist
    _FAC = fac
    _ALLOWED_TOKENS = sorted(allowed_tokens_set)
    _LOADED = True


def get_allowed_facility_tokens() -> List[str]:
    _ensure_loaded()
    return list(_ALLOWED_TOKENS)


def _score_candidate(
    dist_km: float,
    urgency: str,
    required_hits: int,
    required_total: int,
    preferred_hits: int,
    beds: Optional[int],
) -> float:
    u = (urgency or "medium").strip().lower()
    if u in {"high", "emergency"}:
        dist_multiplier = 3.0
        emergency = True
    elif u == "low":
        dist_multiplier = 1.2
        emergency = False
    else:
        dist_multiplier = 2.0
        emergency = False

    dist_penalty = dist_km * dist_multiplier

    req_score = 0.0
    if required_total > 0:
        req_score = (required_hits / required_total) * 300.0

    pref_score = preferred_hits * 80.0
    dist_score = -dist_penalty * 25.0

    beds_bonus = 0.0
    if beds is not None and not emergency:
        beds_bonus = min(beds, 500) * 0.08

    return req_score + pref_score + dist_score + beds_bonus


def _collect_scored(
    origin_kode: str,
    maxd: float,
    urgency: str,
    required: List[str],
    preferred: List[str],
    require_all_required: bool,
    require_distance: bool,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for h in _HOSPITALS:
        if h.kode == origin_kode:
            continue

        d = _DIST.get((origin_kode, h.kode))
        if d is None:
            if require_distance:
                continue
            dist_for_score = 999.0
            distance_km = None
        else:
            if d > maxd:
                continue
            dist_for_score = float(d)
            distance_km = float(d)

        finfo = _FAC.get(h.kode, FacilityInfo(set(), None))
        fac_set = finfo.facilities

        matched_required = [t for t in required if t in fac_set]
        missing_required = [t for t in required if t not in fac_set]
        required_hits = len(matched_required)
        required_total = len(required)

        if require_all_required and required_total > 0 and required_hits < required_total:
            continue

        matched_pref = [t for t in preferred if t in fac_set]
        preferred_hits = len(matched_pref)

        score = _score_candidate(
            dist_km=dist_for_score,
            urgency=urgency,
            required_hits=required_hits,
            required_total=required_total,
            preferred_hits=preferred_hits,
            beds=finfo.beds,
        )

        scored.append({
            "score": score,
            "hospital": h,
            "distance_km": distance_km,
            "matched_required": matched_required,
            "missing_required": missing_required,
            "matched_preferred": matched_pref,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def build_recommendation_payload(origin_kode: str, origin_name: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_loaded()

    urgency = str(criteria.get("urgency", "medium")).strip().lower()
    required: List[str] = [str(x) for x in (criteria.get("required_facilities") or [])]
    preferred: List[str] = [str(x) for x in (criteria.get("preferred_facilities") or [])]
    maxd_initial = float(criteria.get("max_distance_km", 12.0))
    safety = str(criteria.get("safety_note", "")).strip()

    if maxd_initial <= 0:
        maxd_initial = 12.0
    if maxd_initial > 100:
        maxd_initial = 100.0

    notes: List[str] = []
    maxd_used = maxd_initial
    relaxed = False

    scored = _collect_scored(origin_kode, maxd_used, urgency, required, preferred, True, True)

    if not scored:
        step = 5.0
        while maxd_used < 100.0 and not scored:
            maxd_used = min(100.0, maxd_used + step)
            scored = _collect_scored(origin_kode, maxd_used, urgency, required, preferred, True, True)
        if maxd_used > maxd_initial and scored:
            notes.append(f"Expanded search radius from {maxd_initial:.1f} km to {maxd_used:.1f} km.")

    if not scored:
        relaxed = True
        scored = _collect_scored(origin_kode, maxd_used, urgency, required, preferred, False, True)
        if scored and required:
            notes.append("No hospital matched all required facilities. Showing the closest best-effort matches.")

    if not scored:
        relaxed = True
        scored = _collect_scored(origin_kode, maxd_used, urgency, required, preferred, False, False)
        if scored:
            notes.append("Distance data is missing for your origin. Results are facility-based (distance unknown).")

    if not scored:
        fallback = next((h for h in _HOSPITALS if h.kode != origin_kode), None)
        if not fallback:
            raise RuntimeError("No hospitals loaded from data1.csv.")
        scored = [{
            "score": 0.0,
            "hospital": fallback,
            "distance_km": None,
            "matched_required": [],
            "missing_required": list(required),
            "matched_preferred": [],
        }]
        notes.append("Fallback used due to missing distance/facility data.")

    best = scored[0]
    alt = scored[1] if len(scored) > 1 else None

    return {
        "origin_name": origin_name,
        "origin_kode": origin_kode,
        "urgency": urgency,
        "required": required,
        "preferred": preferred,
        "max_distance_requested": maxd_initial,
        "max_distance_used": maxd_used,
        "relaxed_required": relaxed,
        "safety_note": safety,
        "notes": notes,
        "best": best,
        "alt": alt,
    }


def _urgency_line_html(urgency: str, safety_note: str) -> str:
    u = (urgency or "medium").strip().lower()
    if u == "high":
        emoji = "🚨"
        label = "High urgency"
        safety = safety_note or "This may be an emergency — go to the nearest ER or call emergency services immediately."
    elif u == "low":
        emoji = "🟢"
        label = "Low urgency"
        safety = safety_note or "Monitor symptoms and seek care if they worsen."
    else:
        emoji = "🟡"
        label = "Medium urgency"
        safety = safety_note or "Seek medical help if symptoms worsen."
    return f"{emoji} <b>{html.escape(label)}</b> — {html.escape(safety)}"


def _format_distance(distance_km: Optional[float]) -> str:
    if distance_km is None:
        return "distance unknown"
    return f"{distance_km:.1f} km"


def render_main_message(payload: Dict[str, Any]) -> str:
    origin_name = str(payload.get("origin_name", "(unknown)"))
    urgency = str(payload.get("urgency", "medium"))
    safety = str(payload.get("safety_note", "")).strip()

    best_h: Hospital = payload["best"]["hospital"]
    best_d = payload["best"]["distance_km"]

    alt = payload.get("alt")

    lines = [
        _urgency_line_html(urgency, safety),
        f"<b>Origin:</b> {html.escape(origin_name)}",
        "",
        f"✅ <b>Best:</b> {html.escape(best_h.nama)} • {_format_distance(best_d)}",
    ]

    if alt:
        alt_h: Hospital = alt["hospital"]
        alt_d = alt["distance_km"]
        lines.append(f"⭐ <b>Alt:</b> {html.escape(alt_h.nama)} • {_format_distance(alt_d)}")

    lines.extend([
        "",
        "➡️ Tap <b>🧠 Reason</b> or type <b>/reason</b> to see why these hospitals were chosen.",
    ])
    return "\n".join(lines)


def render_reason_message(payload: Dict[str, Any]) -> str:
    required = list(payload.get("required", []) or [])
    preferred = list(payload.get("preferred", []) or [])
    maxd_req = payload.get("max_distance_requested", 0.0)

    best = payload["best"]
    alt = payload.get("alt")
    notes = payload.get("notes", []) or []

    def _block_for(label: str, item: Dict[str, Any]) -> str:
        h: Hospital = item["hospital"]
        matched_pref = item.get("matched_preferred", []) or []
        missing_req = item.get("missing_required", []) or []
        parts = [f"<b>{html.escape(label)} — {html.escape(h.nama)}</b>"]
        parts.append(f"Matched: {html.escape(_format_token_list(list(matched_pref)))}")
        if required:
            parts.append(f"Missing: {html.escape(_format_token_list(list(missing_req)))}")
        return "\n".join(parts)

    lines = [
        "🧠 <b>Reasoning Details</b>",
        "",
        "<b>Criteria</b>",
        f"Required: {html.escape(_format_token_list(required))}",
        f"Preferred: {html.escape(_format_token_list(preferred))}",
        "",
        _block_for("Best", best),
    ]

    if alt:
        lines.extend(["", _block_for("Alt", alt)])

    note_lines = []
    if maxd_req:
        note_lines.append(f"• Max distance requested: {float(maxd_req):.1f} km")
    if notes:
        note_lines.extend([f"• {n}" for n in notes])

    if note_lines:
        lines.extend(["", "<b>Notes</b>", *[html.escape(x) for x in note_lines]])

    return "\n".join(lines)
