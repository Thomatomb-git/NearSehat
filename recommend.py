from __future__ import annotations

import csv
import os
import re
import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI
from telegram import Update
from telegram.ext import CommandHandler, ContextTypes

# ============================================================
# NearSehat - /recommend
#
# Requirements dari kamu:
# - data1, data2, data3 ada di folder "data/" sejajar bot.py
# - origin disimpan sementara (context.user_data["origin_hospital"])
# - hasil /details disimpan sementara (context.user_data["llm_result"], "complaint")
# - /recommend pakai AI (gpt-5-mini) untuk menyusun kriteria rujukan
# - selection & ranking RS deterministic pakai CSV (anti-halusinasi)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default model sesuai request kamu.
# Kalau mau override: set env OPENAI_MODEL=gpt-5-mini (atau model lain).
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-5-mini"


# -----------------------------
# Data model RS dari data1
# -----------------------------
@dataclass(frozen=True)
class Hospital:
    kode: str
    nama: str
    jenis: str
    kelas: str
    alamat: str


# -----------------------------
# Helper path (data folder)
# -----------------------------
def _candidate_data_paths(filename_no_ext: str) -> List[Path]:
    """
    Cari file di:
      - <folder script ini>/data/<name>.csv
      - <folder script ini>/data/<name>
      - <cwd>/data/<name>.csv
      - <cwd>/data/<name>
    plus optional env override:
      - DATA_DIR
      - DATA1_PATH / DATA2_PATH / DATA3_PATH (langsung ke file)
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    paths: List[Path] = []

    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        d = Path(env_dir)
        paths.append(d / f"{filename_no_ext}.csv")
        paths.append(d / filename_no_ext)

    # default (sefolder bot.py biasanya)
    paths.append(here / "data" / f"{filename_no_ext}.csv")
    paths.append(here / "data" / filename_no_ext)

    # fallback (kalau run dari folder bot.py)
    paths.append(cwd / "data" / f"{filename_no_ext}.csv")
    paths.append(cwd / "data" / filename_no_ext)

    # De-dup preserve order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        sp = str(p)
        if sp not in seen:
            uniq.append(p)
            seen.add(sp)
    return uniq


def _find_file_or_raise(name_no_ext: str, env_key: Optional[str] = None) -> Path:
    """
    Cari file dataX. Kalau env_key diset dan env var-nya ada, pakai itu.
    """
    if env_key:
        envp = os.getenv(env_key)
        if envp and Path(envp).exists():
            return Path(envp)

    for p in _candidate_data_paths(name_no_ext):
        if p.exists():
            return p

    tried = "\n".join([f"- {p}" for p in _candidate_data_paths(name_no_ext)])
    raise FileNotFoundError(
        f"Cannot find {name_no_ext}.\n"
        f"Expected: data/{name_no_ext}.csv\n\nTried:\n{tried}"
    )


# -----------------------------
# Normalisasi token fasilitas
# -----------------------------
def _norm_token(s: str) -> str:
    """
    Ubah nama kolom fasilitas jadi token stabil:
      "Emergency room (24 jam) " -> "EMERGENCY_ROOM_24_JAM"
      "CT-Scan" -> "CT_SCAN"
      "Stroke Unit" -> "STROKE_UNIT"
    """
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


# -----------------------------
# Load data1 (RS list)
# -----------------------------
def load_data1() -> List[Hospital]:
    p = _find_file_or_raise("data1", env_key="DATA1_PATH")
    hospitals: List[Hospital] = []

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_l = {str(k).strip().lower(): v for k, v in (row or {}).items()}

            kode = str(row_l.get("kode", "")).strip()
            nama = str(row_l.get("nama", "")).strip()
            jenis = str(row_l.get("jenis", "")).strip()
            kelas = str(row_l.get("kelas", "")).strip()
            alamat = str(row_l.get("alamat", "")).strip()

            if not kode or not nama:
                continue

            hospitals.append(Hospital(kode=kode, nama=nama, jenis=jenis, kelas=kelas, alamat=alamat))

    return hospitals


# -----------------------------
# Load data2 (distance: origin->target)
# Expected format (dari file kamu yang dulu):
#   origin,target,distance
# -----------------------------
def load_data2_distances() -> Dict[Tuple[str, str], float]:
    p = _find_file_or_raise("data2", env_key="DATA2_PATH")

    dist: Dict[Tuple[str, str], float] = {}
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]

        # Format yang kita dukung:
        # - origin/target/distance
        if not (("origin" in headers) and ("target" in headers) and ("distance" in headers)):
            raise ValueError(
                "data2 format not recognized. Expected headers: origin,target,distance"
            )

        for row in reader:
            row_l = {str(k).strip().lower(): v for k, v in (row or {}).items()}
            o = str(row_l.get("origin", "")).strip()
            t = str(row_l.get("target", "")).strip()
            d_raw = str(row_l.get("distance", "")).strip()

            if not o or not t or not d_raw:
                continue

            try:
                d = float(d_raw)
            except:
                continue

            dist[(o, t)] = d

    # Optional: jarak ke diri sendiri = 0
    for (o, t) in list(dist.keys()):
        dist[(o, o)] = 0.0

    return dist


# -----------------------------
# Load data3 (fasilitas/spesialis)
# Strategy:
# - Setiap kolom selain kode/nama dianggap fitur.
# - value truthy (1/true/yes) -> masuk fasilitas
# - "Total beds" disimpan sebagai capacity (kalau ada)
# -----------------------------
@dataclass
class FacilityInfo:
    facilities: Set[str]
    beds: Optional[int]


def _truthy(v: str) -> bool:
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "ya", "iya", "available", "ada")


def load_data3_facilities() -> Tuple[Dict[str, FacilityInfo], Set[str]]:
    p = _find_file_or_raise("data3", env_key="DATA3_PATH")

    by_kode: Dict[str, FacilityInfo] = {}
    all_tokens: Set[str] = set()

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = [c for c in (reader.fieldnames or [])]

        # Cari kolom kode (wajib)
        # Kita toleran: "kode" / "Kode" / dst.
        code_col = None
        name_col = None
        for c in fields:
            cl = c.strip().lower()
            if cl == "kode":
                code_col = c
            if cl == "nama":
                name_col = c
        if not code_col:
            raise ValueError("data3 must contain 'kode' column")

        for row in reader:
            kode = str(row.get(code_col, "")).strip()
            if not kode:
                continue

            facilities: Set[str] = set()
            beds: Optional[int] = None

            for col in fields:
                if col == code_col or (name_col and col == name_col):
                    continue

                val = row.get(col, "")
                token = _norm_token(col)

                # Total beds -> numeric feature
                if token in ("TOTAL_BEDS", "TOTAL_BED", "BEDS"):
                    try:
                        beds = int(float(str(val).strip()))
                    except:
                        beds = beds
                    continue

                # boolean facility/specialty
                if _truthy(val):
                    facilities.add(token)
                    all_tokens.add(token)

            by_kode[kode] = FacilityInfo(facilities=facilities, beds=beds)

    return by_kode, all_tokens


# ============================================================
# AI step: translate details -> criteria JSON (required/preferred/max_distance)
#
# Penting:
# - kita pakai JSON mode di Responses API:
#     text={"format": {"type": "json_object"}}
# - prompt harus mengandung kata "JSON" (rule dari JSON mode)
# ============================================================
def _criteria_system_prompt() -> str:
    return (
        "You are a hospital referral criteria generator designed to output JSON only. "
        "You will receive: (1) user's complaint, (2) triage JSON (services, urgency), "
        "and (3) a list of allowed facility tokens. "
        "Output a STRICT JSON object with keys:\n"
        "required_facilities (array of strings),\n"
        "preferred_facilities (array of strings),\n"
        "max_distance_km (number),\n"
        "safety_note (string, one sentence).\n"
        "Use ONLY tokens from the allowed list. Return JSON only."
    )


def _criteria_user_prompt(complaint: str, triage: dict, allowed_tokens: List[str]) -> str:
    # Kita kasih token yang boleh dipakai supaya model nggak ngarang fasilitas.
    # Karena dataset kamu kecil, ini aman.
    return (
        "Complaint:\n"
        f"{complaint}\n\n"
        f"Triage JSON:\n{json.dumps(triage, ensure_ascii=False)}\n\n"
        "Allowed facility tokens:\n"
        f"{allowed_tokens}\n\n"
        "Rules:\n"
        "- If urgency is high, prioritize Emergency room and ICU related needs.\n"
        "- required_facilities should be minimal but necessary.\n"
        "- preferred_facilities are nice-to-have.\n"
        "- max_distance_km should be smaller for higher urgency.\n"
        "Remember : always output english despite prompt is not in english\n"
        "Remember: output JSON only."
    )


async def _call_llm_criteria(complaint: str, triage: dict, allowed_tokens: Set[str]) -> Tuple[bool, dict]:
    """
    Call gpt-5-mini via Responses API, JSON mode.
    """
    if not OPENAI_API_KEY:
        return False, {"error": "OPENAI_API_KEY is not set."}

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Batasi token list biar prompt nggak bengkak
    allowed_sorted = sorted(list(allowed_tokens))
    allowed_sorted = allowed_sorted[:500]  # dataset kamu kecil, ini biasanya sudah cukup

    def _req():
        # JSON mode: text.format json_object (lihat docs)
        # https://platform.openai.com/docs/guides/structured-outputs#json-mode
        return client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": _criteria_system_prompt()},
                {
                    "role": "user",
                    "content": _criteria_user_prompt(complaint, triage, allowed_sorted),
                },
            ],
            text={"format": {"type": "json_object"}},
        )

    try:
        rsp = await asyncio.get_running_loop().run_in_executor(None, _req)
        raw = (rsp.output_text or "").strip()
        if not raw:
            return False, {"error": "Empty model output."}

        data = json.loads(raw)
        if not isinstance(data, dict):
            return False, {"error": "Model returned non-object JSON."}

        # Normalize & sanitize
        req = data.get("required_facilities", [])
        pref = data.get("preferred_facilities", [])

        if not isinstance(req, list): req = []
        if not isinstance(pref, list): pref = []

        req_norm = []
        for x in req:
            t = _norm_token(str(x))
            if t in allowed_tokens:
                req_norm.append(t)

        pref_norm = []
        for x in pref:
            t = _norm_token(str(x))
            if t in allowed_tokens and t not in req_norm:
                pref_norm.append(t)

        maxd = data.get("max_distance_km", 12)
        try:
            maxd = float(maxd)
        except:
            maxd = 12.0

        safety_note = str(data.get("safety_note", "")).strip()
        if not safety_note:
            safety_note = "If symptoms are severe, go to the nearest emergency room immediately."

        return True, {
            "required_facilities": req_norm,
            "preferred_facilities": pref_norm,
            "max_distance_km": maxd,
            "safety_note": safety_note,
        }

    except Exception as e:
        return False, {"error": f"LLM error: {e}"}


# ============================================================
# Deterministic fallback (kalau AI gagal)
# ============================================================
def _fallback_criteria(triage: dict, allowed_tokens: Set[str]) -> dict:
    """
    Kalau AI down, kita tetap bisa jalan pakai mapping sederhana.
    triage berasal dari /details (services + urgency).
    """
    urgency = str(triage.get("urgency", "medium")).strip().lower()
    services = triage.get("services", [])
    if not isinstance(services, list):
        services = []

    # Map service -> token data3 (berdasarkan kolom di data3 kamu)
    # Token ini harus match _norm_token(col) dari data3.
    service_map = {
        "emergency": ["EMERGENCY_ROOM_24_JAM"],
        "pulmonology": ["PULMONOLOGY"],
        "cardiology": ["CARDIOLOGY"],
        "neurology": ["NEUROLOGY"],
        "orthopedics": ["ORTHOPEDICS"],
        "obstetrics_gynecology": ["OBSTETRICS_GYNECOLOGY"],
        "pediatrics": ["PEDIATRICS"],
        "cathlab": ["CATH_LAB"],
        "ct_scan_24h": ["CT_SCAN"],
        "stroke_unit": ["STROKE_UNIT", "CT_SCAN"],
    }

    required: List[str] = []
    preferred: List[str] = []

    for s in services:
        key = str(s).strip().lower()
        for tok in service_map.get(key, []):
            if tok in allowed_tokens and tok not in required:
                required.append(tok)

    # Urgency-based requirements
    if urgency in ("high", "emergency"):
        # Pastikan ER + ICU jika tersedia tokennya
        if "EMERGENCY_ROOM_24_JAM" in allowed_tokens and "EMERGENCY_ROOM_24_JAM" not in required:
            required.insert(0, "EMERGENCY_ROOM_24_JAM")
        if "ICU" in allowed_tokens and "ICU" not in required:
            required.append("ICU")

    # Preferred goodies
    if "CT_SCAN" in allowed_tokens and "CT_SCAN" not in required:
        preferred.append("CT_SCAN")

    # Max distance default
    maxd = 20.0
    if urgency in ("high", "emergency"):
        maxd = 10.0
    elif urgency == "medium":
        maxd = 15.0

    return {
        "required_facilities": required,
        "preferred_facilities": preferred,
        "max_distance_km": maxd,
        "safety_note": "If symptoms are severe, go to the nearest emergency room immediately.",
    }


# ============================================================
# Ranking
# ============================================================
def _score_candidate(
    *,
    dist_km: float,
    required_hits: int,
    required_total: int,
    preferred_hits: int,
    beds: Optional[int],
    urgency: str,
) -> float:
    """
    Score lebih tinggi = lebih bagus.

    Ide:
    - Jarak lebih dekat selalu bagus (apalagi urgent)
    - Preferred facilities nambah poin
    - Bed capacity sedikit bantu untuk non-emergency (opsional)
    """
    # Urgency weight: makin urgent, penalti jarak makin besar
    urgency = (urgency or "medium").lower()
    dist_penalty = dist_km
    if urgency in ("high", "emergency"):
        dist_penalty = dist_km * 3.0
    elif urgency == "medium":
        dist_penalty = dist_km * 2.0
    else:
        dist_penalty = dist_km * 1.2

    score = 0.0

    # Required pass is enforced outside; but we still reward “complete match”
    if required_total > 0:
        score += (required_hits / required_total) * 300.0

    score += preferred_hits * 80.0
    score -= dist_penalty * 25.0

    # Beds: small nudge
    if beds is not None and urgency not in ("high", "emergency"):
        score += min(beds, 500) * 0.08

    return score


# ============================================================
# Main handler: /recommend
# ============================================================
async def recommend_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /recommend:
    - cek origin sudah diset
    - cek /details sudah ada
    - load data1/data2/data3
    - call AI untuk criteria (required/preferred/max_distance)
    - rank RS deterministically
    - output Best + Alternative
    """
    msg = update.message
    if not msg:
        return

    # 1) prerequisite: origin
    origin = context.user_data.get("origin_hospital")
    if not origin or not isinstance(origin, dict) or not origin.get("kode"):
        await msg.reply_text(
            "Origin hospital is not set yet.\n"
            "Please use /setorigin first to choose the hospital you are currently at."
        )
        return

    origin_kode = str(origin["kode"]).strip()

    # 2) prerequisite: details
    triage = context.user_data.get("llm_result")
    complaint = context.user_data.get("complaint")
    if not triage or not isinstance(triage, dict) or not complaint:
        await msg.reply_text(
            "No /details result found yet.\n"
            "Please use /details first, then run /recommend."
        )
        return

    # 3) load datasets
    try:
        hospitals = load_data1()
        dist = load_data2_distances()
        fac_by_kode, allowed_tokens = load_data3_facilities()
    except Exception as e:
        await msg.reply_text(f"Error loading CSV data: {e}")
        return

    # 4) AI criteria
    await msg.reply_text("Analyzing referral criteria with AI ...")

    ok, criteria = await _call_llm_criteria(str(complaint), triage, allowed_tokens)
    if not ok:
        # fallback deterministic (biar bot tetap jalan)
        criteria = _fallback_criteria(triage, allowed_tokens)

    required = criteria.get("required_facilities", [])
    preferred = criteria.get("preferred_facilities", [])
    maxd = float(criteria.get("max_distance_km", 12.0))
    safety_note = str(criteria.get("safety_note", "")).strip()

    urgency = str(triage.get("urgency", "medium")).strip().lower()

    # 5) build candidate list
    scored: List[Tuple[float, Hospital, float, int, int, int]] = []
    for h in hospitals:
        if h.kode == origin_kode:
            continue

        d = dist.get((origin_kode, h.kode))
        if d is None:
            continue

        # distance filter
        if d > maxd:
            continue

        finfo = fac_by_kode.get(h.kode)
        facilities = finfo.facilities if finfo else set()
        beds = finfo.beds if finfo else None

        # required check (hard filter)
        required_total = len(required)
        required_hits = sum(1 for t in required if t in facilities)

        if required_total > 0 and required_hits < required_total:
            continue

        preferred_hits = sum(1 for t in preferred if t in facilities)

        score = _score_candidate(
            dist_km=d,
            required_hits=required_hits,
            required_total=max(required_total, 1),
            preferred_hits=preferred_hits,
            beds=beds,
            urgency=urgency,
        )

        scored.append((score, h, d, required_hits, required_total, preferred_hits))

    if not scored:
        await msg.reply_text(
            "No hospitals matched your criteria within the distance limit.\n"
            f"- Max distance: {maxd} km\n"
            f"- Required: {required}\n\n"
            "Try running /details again with a more detailed description, or set a different origin."
        )
        return

    scored.sort(key=lambda x: x[0], reverse=True)

    best = scored[0]
    alt = scored[1] if len(scored) >= 2 else None

    def _format_choice(tag: str, item: Tuple[float, Hospital, float, int, int, int]) -> str:
        _, h, d, rh, rt, ph = item
        finfo = fac_by_kode.get(h.kode)
        facilities = finfo.facilities if finfo else set()

        matched_req = [t for t in required if t in facilities]
        matched_pref = [t for t in preferred if t in facilities]

        lines = []
        lines.append(f"{tag}")
        lines.append(f"- {h.nama} ({h.kode})")
        lines.append(f"- Distance from origin: {d:.1f} km")
        lines.append(f"- Address: {h.alamat}")
        if finfo and finfo.beds is not None:
            lines.append(f"- Beds: {finfo.beds}")

        # Alasan data-based
        if required:
            lines.append(f"- Required match: {len(matched_req)}/{len(required)} -> {matched_req}")
        if preferred:
            lines.append(f"- Preferred match: {len(matched_pref)}/{len(preferred)} -> {matched_pref}")

        return "\n".join(lines)

    # 6) output
    header = []
    header.append("🏥 NearSehat Recommendation")
    header.append(f"Origin: {origin.get('nama', '-') } ({origin_kode})")
    header.append(f"Urgency: {urgency}")
    header.append(f"Criteria: max_distance_km={maxd}")
    if required:
        header.append(f"Required: {required}")
    if preferred:
        header.append(f"Preferred: {preferred}")
    header.append("")

    out = []
    out.append("\n".join(header))
    out.append(_format_choice("✅ Best", best))
    out.append("")
    if alt:
        out.append(_format_choice("⭐ Alternative", alt))
        out.append("")
    if safety_note:
        out.append(f"⚠️ Note: {safety_note}")

    # Simpan hasil terakhir (sementara) biar gampang debug
    context.user_data["last_recommendation"] = {
        "origin": origin_kode,
        "criteria": criteria,
        "best": {"kode": best[1].kode, "distance": best[2]},
        "alternative": {"kode": alt[1].kode, "distance": alt[2]} if alt else None,
    }

    await msg.reply_text("\n".join(out))


def get_recommend_handler():
    """
    Di bot.py:
        from recommend import get_recommend_handler
        app.add_handler(get_recommend_handler())
    """
    return CommandHandler("recommend", recommend_handler)
