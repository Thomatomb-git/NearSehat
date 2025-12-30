from __future__ import annotations

# setorigin.py (helpers)
# Modul ini berisi logic pencarian RS + inline keyboard.
# Tidak lagi expose command /setorigin. Flow-nya di-handle oleh bot.py.

import csv
import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# =========================
# Conversation State
# =========================
ORIGIN_QUERY = 1

# Callback prefix untuk inline button
CALLBACK_PREFIX_ORIGIN = "origin:"


@dataclass(frozen=True)
class Hospital:
    kode: str
    nama: str
    jenis: str
    kelas: str
    alamat: str


DEFAULT_DATA1_NAME = "data1.csv"


def _resolve_data1_path() -> Path:
    p = os.getenv("DATA1_PATH")
    if p:
        return Path(p).expanduser()

    data_dir = os.getenv("DATA_DIR")
    if data_dir:
        return (Path(data_dir).expanduser() / DEFAULT_DATA1_NAME)

    candidates = [
        Path.cwd() / DEFAULT_DATA1_NAME,
        Path.cwd() / "data" / DEFAULT_DATA1_NAME,
    ]

    here = Path(__file__).resolve().parent
    candidates.extend([
        here / DEFAULT_DATA1_NAME,
        here / "data" / DEFAULT_DATA1_NAME,
    ])

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]


_HOSPITALS: List[Hospital] = []
_BY_KODE: Dict[str, Hospital] = {}
_NORM_NAME: List[str] = []
_NORM_TOKENS: List[List[str]] = []
_LOADED: bool = False


def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ensure_loaded() -> None:
    global _LOADED, _HOSPITALS, _BY_KODE, _NORM_NAME, _NORM_TOKENS
    if _LOADED:
        return

    path = _resolve_data1_path()
    if not path.exists():
        raise FileNotFoundError(
            f"data1.csv not found. Tried: {path}. "
            "Set DATA1_PATH or DATA_DIR to fix."
        )

    hospitals: List[Hospital] = []
    by_kode: Dict[str, Hospital] = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"kode", "nama", "jenis", "kelas", "alamat"}
        if not reader.fieldnames:
            raise ValueError("data1.csv has no header row.")
        missing = required_cols - set([h.strip() for h in reader.fieldnames])
        if missing:
            raise ValueError(f"data1.csv missing columns: {sorted(missing)}")

        for row in reader:
            kode = str(row.get("kode", "")).strip()
            if not kode:
                continue
            hosp = Hospital(
                kode=kode,
                nama=str(row.get("nama", "")).strip(),
                jenis=str(row.get("jenis", "")).strip(),
                kelas=str(row.get("kelas", "")).strip(),
                alamat=str(row.get("alamat", "")).strip(),
            )
            hospitals.append(hosp)
            by_kode[kode] = hosp

    _HOSPITALS = hospitals
    _BY_KODE = by_kode
    _NORM_NAME = [_normalize_text(h.nama) for h in _HOSPITALS]
    _NORM_TOKENS = [n.split() for n in _NORM_NAME]
    _LOADED = True


def get_hospital_by_kode(kode: str) -> Optional[Hospital]:
    _ensure_loaded()
    return _BY_KODE.get(str(kode).strip())


def search_hospitals(query: str, limit: int = 10) -> List[Hospital]:
    """
    Search RS by kode or name with scoring:
    - token hits
    - substring boost
    - fuzzy ratio (full name)
    - fuzzy ratio against individual tokens (fix typo like 'ukirda' vs 'ukrida')
    """
    _ensure_loaded()

    q_raw = (query or "").strip()
    if not q_raw:
        return []

    q = _normalize_text(q_raw)
    if not q:
        return []

    if q_raw.isdigit():
        exact = _BY_KODE.get(q_raw)
        results: List[Hospital] = []
        if exact:
            results.append(exact)

        for hosp in _HOSPITALS:
            if hosp.kode != q_raw and q_raw in hosp.kode:
                results.append(hosp)
                if len(results) >= limit:
                    break
        return results[:limit]

    q_tokens = set(q.split())

    scored = []
    for hosp, norm_name, name_tokens in zip(_HOSPITALS, _NORM_NAME, _NORM_TOKENS):
        score = 0.0
        name_token_set = set(name_tokens)
        token_hits = len(q_tokens & name_token_set)
        score += token_hits * 120.0

        if q in norm_name:
            pos = norm_name.find(q)
            score += 180.0 + max(0.0, 40.0 - 0.5 * pos)

        ratio_full = difflib.SequenceMatcher(None, q, norm_name).ratio()

        ratio_tok = 0.0
        if name_tokens:
            ratio_tok = max(difflib.SequenceMatcher(None, q, t).ratio() for t in name_tokens)

        ratio = max(ratio_full, ratio_tok)

        if token_hits > 0 or q in norm_name or ratio >= 0.58:
            score += ratio * 100.0
            scored.append((score, hosp))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:limit]]


def build_origin_keyboard(hospitals: List[Hospital]) -> InlineKeyboardMarkup:
    buttons: List[InlineKeyboardButton] = []
    for h in hospitals:
        label = f"{h.nama} ({h.kode})"
        buttons.append(InlineKeyboardButton(label, callback_data=f"{CALLBACK_PREFIX_ORIGIN}{h.kode}"))

    rows = []
    row = []
    for b in buttons:
        row.append(b)
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    return InlineKeyboardMarkup(rows)
