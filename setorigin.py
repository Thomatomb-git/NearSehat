from __future__ import annotations

import csv
import os
import re
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ConversationHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# =========================
# Conversation State
# =========================
# Setelah user mengetik /setorigin, bot masuk ke state ini:
# - Menunggu user mengetik keyword RS / kode RS
# - Atau user klik tombol hasil pencarian (callback query)
ORIGIN_QUERY = 1


# =========================
# Data model untuk 1 RS
# =========================
@dataclass(frozen=True)
class Hospital:
    kode: str
    nama: str
    jenis: str
    kelas: str
    alamat: str

    @property
    def label(self) -> str:
        """
        Label tombol (inline keyboard). Dibikin ringkas supaya gampang discan.
        """
        # Kalau namanya kepanjangan, potong biar tombol enak dilihat
        name = self.nama
        if len(name) > 42:
            name = name[:39].rstrip() + "..."
        return f"{name} ({self.kode})"


# =========================
# Helper: normalisasi string
# =========================
def _norm(s: str) -> str:
    """
    Normalisasi teks input user dan teks nama RS:
    - lowercase
    - buang tanda baca
    - rapihin whitespace
    """
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # keep alnum + space
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# Lokasi file data1
# =========================
def _candidate_data1_paths() -> List[Path]:
    """
    Aturan yang kamu minta:
    - File CSV selalu ada di folder anak bernama: "data"
    - Folder "data" selalu sejajar dengan bot.py
    - Karena setorigin.py biasanya se-folder dengan bot.py, kita pakai:
        base_dir = folder script ini
      dan juga fallback:
        base_dir = current working directory (kalau kamu run dari folder bot.py)

    Nama file:
      - data1.csv (umum)
      - data1 (kalau kamu beneran simpan tanpa ekstensi)

    Optional override (kalau suatu saat kamu mau):
      - env var DATA1_PATH: path langsung ke file data1
      - env var DATA_DIR: path folder data (mis: C:/data)
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    paths: List[Path] = []

    # Override path langsung kalau diset
    env_data1 = os.getenv("DATA1_PATH")
    if env_data1:
        paths.append(Path(env_data1))

    # Override folder data kalau diset
    env_dir = os.getenv("DATA_DIR")
    if env_dir:
        d = Path(env_dir)
        paths.append(d / "data1.csv")
        paths.append(d / "data1")

    # Default: subfolder data di folder bot.py (anggap sefolder dengan setorigin.py)
    paths.append(here / "data" / "data1.csv")
    paths.append(here / "data" / "data1")

    # Fallback: subfolder data di current working directory
    paths.append(cwd / "data" / "data1.csv")
    paths.append(cwd / "data" / "data1")

    # De-dup preserve order
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        ps = str(p)
        if ps not in seen:
            uniq.append(p)
            seen.add(ps)
    return uniq


def load_hospitals_from_data1() -> List[Hospital]:
    """
    Load data RS dari data/data1(.csv).
    Expected kolom: kode, nama, jenis, kelas, alamat
    (case-insensitive; misal Kode / NAMA juga diterima).
    """
    for p in _candidate_data1_paths():
        if p.exists():
            hospitals: List[Hospital] = []
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Buat mapping key jadi lowercase biar robust terhadap variasi header
                    row_l = {str(k).strip().lower(): v for k, v in (row or {}).items()}

                    kode = str(row_l.get("kode", "")).strip()
                    nama = str(row_l.get("nama", "")).strip()
                    jenis = str(row_l.get("jenis", "")).strip()
                    kelas = str(row_l.get("kelas", "")).strip()
                    alamat = str(row_l.get("alamat", "")).strip()

                    # Kode selalu numeric (kata kamu), tapi kita simpan sebagai string
                    if not kode or not nama:
                        continue

                    hospitals.append(Hospital(kode=kode, nama=nama, jenis=jenis, kelas=kelas, alamat=alamat))

            return hospitals

    tried = "\n".join([f"- {p}" for p in _candidate_data1_paths()])
    raise FileNotFoundError(
        "Cannot find data1 CSV.\n"
        "Please make sure your folder structure is like:\n"
        "  <project>\\bot.py\n"
        "  <project>\\setorigin.py\n"
        "  <project>\\data\\data1.csv\n\n"
        "Tried:\n" + tried
    )


# =========================
# Cache dataset (load sekali)
# =========================
# Kita load di import-time supaya query cepat.
# Kalau gagal load, bot tidak crash, tapi /setorigin akan menolak dengan pesan error.
_LOAD_ERROR: Optional[str] = None
try:
    _HOSPITALS: List[Hospital] = load_hospitals_from_data1()
except Exception as e:
    _HOSPITALS = []
    _LOAD_ERROR = str(e)

# Index by kode -> Hospital (lookup O(1))
_BY_KODE: Dict[str, Hospital] = {h.kode: h for h in _HOSPITALS}

# Precompute nama yang sudah dinormalisasi (biar search gak norm berulang)
_NORM_NAME: Dict[str, str] = {h.kode: _norm(h.nama) for h in _HOSPITALS}


# =========================
# Fuzzy scoring function
# =========================
def _fuzzy_ratio(a: str, b: str) -> float:
    """
    Return similarity ratio [0..1] using stdlib difflib.
    (No extra dependency needed.)
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


def _search_hospitals(query: str, limit: int = 10) -> List[Hospital]:
    """
    Cari RS berdasarkan input user.

    Strategy:
    1) Kalau query numeric -> treat as kode:
       - exact match => top result
       - partial match => show candidates
    2) Kalau query teks -> kombinasi:
       - substring match (kuat)
       - token match (berapa kata yang kena)
       - fuzzy match (untuk typo / beda spasi)

    Output: top `limit` hospitals.
    """
    q_raw = (query or "").strip()
    q = _norm(q_raw)
    if not q:
        return []

    # -------------------------
    # Case 1: query numeric
    # -------------------------
    if q.isdigit():
        # Exact kode hit
        if q in _BY_KODE:
            return [_BY_KODE[q]]

        # Partial kode match (misal user ketik 3174)
        hits = [h for h in _HOSPITALS if q in h.kode]
        return hits[:limit]

    # -------------------------
    # Case 2: query teks (nama)
    # -------------------------
    q_tokens = q.split()

    scored: List[Tuple[float, Hospital]] = []
    for h in _HOSPITALS:
        hn = _NORM_NAME.get(h.kode, "")
        if not hn:
            continue

        # Basic signals:
        # - token_hits: berapa token query ada di nama RS
        token_hits = sum(1 for t in q_tokens if t and t in hn)

        # - substring bonus: query string utuh muncul di nama RS?
        substr_pos = hn.find(q)
        substr_hit = substr_pos != -1

        # - fuzzy ratio: handle typo kecil (cengkarng vs cengkareng)
        #   NOTE: difflib cenderung lebih bagus kalau panjang string mirip,
        #   tapi untuk query yang pendek masih cukup membantu.
        ratio = _fuzzy_ratio(q, hn)

        # Filter ringan biar gak semua RS masuk:
        # - Kalau query sangat pendek (<=2 char), wajib substring/token match
        if len(q) <= 2 and not (token_hits > 0 or substr_hit):
            continue

        # - Kalau query normal, terima jika:
        #   ada token hit, atau substring hit, atau fuzzy cukup tinggi
        if not (token_hits > 0 or substr_hit or ratio >= 0.58):
            continue

        # Scoring:
        # - token_hits paling kuat
        # - substring bonus cukup besar
        # - fuzzy ratio sebagai tambahan
        # - substring pos (lebih awal => lebih relevan)
        score = 0.0
        score += token_hits * 120.0
        if substr_hit:
            score += 180.0
            # Semakin awal muncul, tambah bonus kecil
            score += max(0.0, 40.0 - (substr_pos * 0.5))
        score += ratio * 100.0

        scored.append((score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:limit]]


# =========================
# Inline keyboard builder
# =========================
def _options_keyboard(options: List[Hospital]) -> InlineKeyboardMarkup:
    """
    Buat keyboard 2 kolom untuk menampilkan opsi RS.
    callback_data = "origin:<kode>"
    """
    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []

    for h in options:
        row.append(InlineKeyboardButton(h.label, callback_data=f"origin:{h.kode}"))
        if len(row) == 2:
            rows.append(row)
            row = []

    if row:
        rows.append(row)

    return InlineKeyboardMarkup(rows)


# =========================
# Handlers: /setorigin entry
# =========================
async def setorigin_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    /setorigin:
    - Pastikan dataset kebaca
    - Beri instruksi ke user
    - Support /setorigin <keyword> langsung juga
    """
    if _LOAD_ERROR or not _HOSPITALS:
        await update.message.reply_text(
            "Dataset can't be read.\n"
            # "Pastikan struktur folder kamu:\n"
            # "  bot.py\n"
            # "  setorigin.py\n"
            # "  data/data1.csv\n\n"
            f"(debug) {_LOAD_ERROR or 'no data loaded'}"
        )
        return ConversationHandler.END

    # Kalau user sudah pernah set origin, kasih info current
    current = context.user_data.get("origin_hospital")
    if current and isinstance(current, dict) and current.get("nama") and current.get("kode"):
        await update.message.reply_text(
            "Your current origin hospital:\n"
            f"- {current['nama']} ({current['kode']})\n\n"
            "Type another hospital's keywords/code to change, or /cancel to cancel."
        )
    else:
        await update.message.reply_text(
            "Set origin hospital.\n\n"
            "Type the hospital's keywords (example: `cengkareng`, `rsud`, `kembangan`) "
            "or type hospital's `code`.\n\n"
            "Click /cancel to cancel."
        )

    # Support: /setorigin <keyword> (misal /setorigin cengkareng)
    args = getattr(context, "args", None) or []
    if args:
        q = " ".join(args)
        options = _search_hospitals(q, limit=10)
        if not options:
            await update.message.reply_text("No match, try another keyword.")
            return ORIGIN_QUERY

        await update.message.reply_text(
            "Select one of the following options:",
            reply_markup=_options_keyboard(options),
        )
        return ORIGIN_QUERY

    return ORIGIN_QUERY


# =========================
# Handler: user kirim keyword
# =========================
async def setorigin_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Terima input teks user, lalu:
    - search top 10
    - tampilkan inline keyboard
    """
    if _LOAD_ERROR or not _HOSPITALS:
        await update.message.reply_text(
            "Dataset unavailable. Please check data/data1.csv."
        )
        return ConversationHandler.END

    text = (update.message.text or "").strip()
    options = _search_hospitals(text, limit=10)

    if not options:
        await update.message.reply_text(
            "No suitable hospital.\n"
            "Try another keyword, example: `rsud`, `cengkareng`, `kalideres`."
        )
        return ORIGIN_QUERY

    await update.message.reply_text(
        "Select your origin hospital from these options:",
        reply_markup=_options_keyboard(options),
    )
    return ORIGIN_QUERY


# =========================
# Handler: user klik tombol
# =========================
async def setorigin_choose(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Callback dari inline keyboard:
    - baca kode RS dari callback_data
    - simpan origin ke context.user_data (sementara)
    """
    query = update.callback_query
    if not query:
        return ConversationHandler.END

    # Wajib answer() supaya UI Telegram gak "loading" terus
    await query.answer()

    data = query.data or ""
    m = re.fullmatch(r"origin:(\d+)", data)
    if not m:
        await query.edit_message_text("Invalid choice. Run /setorigin again.")
        return ConversationHandler.END

    kode = m.group(1)
    h = _BY_KODE.get(kode)
    if not h:
        await query.edit_message_text("Hospital not found. Run /setorigin again.")
        return ConversationHandler.END

    # Simpan origin sementara untuk dipakai /recommend
    context.user_data["origin_hospital"] = {
        "kode": h.kode,
        "nama": h.nama,
        "jenis": h.jenis,
        "kelas": h.kelas,
        "alamat": h.alamat,
    }

    await query.edit_message_text(
        "✅ Origin hospital successfully set!\n"
        f"- {h.nama} ({h.kode})\n"
        f"- {h.alamat}\n\n"
        "Next: you can click /details."
    )

    return ConversationHandler.END


# =========================
# Handler: cancel
# =========================
async def setorigin_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Set origin cancelled.")
    return ConversationHandler.END


# =========================
# Export function: get handler
# =========================
def get_setorigin_handler() -> ConversationHandler:
    """
    Di bot.py, cukup:
        app.add_handler(get_setorigin_handler())

    ConversationHandler state:
      - ORIGIN_QUERY:
          * MessageHandler => setorigin_search (user ngetik keyword)
          * CallbackQueryHandler => setorigin_choose (user klik tombol)
    """
    return ConversationHandler(
        entry_points=[CommandHandler("setorigin", setorigin_entry)],
        states={
            ORIGIN_QUERY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, setorigin_search),
                CallbackQueryHandler(setorigin_choose, pattern=r"^origin:\d+$"),
            ],
        },
        fallbacks=[CommandHandler("cancel", setorigin_cancel)],
        name="setorigin_conversation",
        persistent=False,
    )
