from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import asyncio
from typing import Tuple

from openai import OpenAI
from telegram import Update
from telegram.ext import (
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")  # default handled below


# =========================
# Helpers
# =========================
def _word_count(s: str) -> int:
    return len([t for t in re.split(r"\s+", (s or "").strip()) if t])


def _origin_is_set(context: ContextTypes.DEFAULT_TYPE) -> Tuple[bool, str]:
    """
    /details wajib setelah /setorigin.
    Kita cek context.user_data['origin_hospital'] yang di-set oleh setorigin.py.
    """
    origin = context.user_data.get("origin_hospital")
    if not origin or not isinstance(origin, dict) or not origin.get("kode"):
        return False, (
            "Origin hospital is not set yet.\n"
            "Please use /setorigin first to choose the hospital you are currently at."
        )
    return True, ""


def _system_prompt() -> str:
    # JSON mode but still: prompt should mention JSON and keys
    return (
        "You are a concise clinical triage helper. "
        "Given a short free-text complaint in Indonesian or English, "
        "output STRICT JSON with keys: "
        "services (array of strings), urgency (low|medium|high), rationale (1 sentence). "
        "Return JSON ONLY. No extra text. Use English."
    )


def _user_prompt(complaint: str) -> str:
    return (
        "Complaint:\n"
        f"{complaint}\n\n"
        "Map to high-level services (e.g., emergency, pulmonology, cardiology, neurology, "
        "orthopedics, obstetrics_gynecology, pediatrics, cathlab, ct_scan_24h, stroke_unit). "
        "If uncertain, choose the closest high-level services. "
        "Set urgency to high if red-flag symptoms suggest immediate care. "
        "Return JSON only."
    )


async def _call_llm(complaint: str) -> Tuple[bool, str]:
    """
    Call gpt-5-mini via Responses API in JSON mode.
    Returns: (ok, payload_or_error)
    """
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY is not set."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        def _req():
            return client.responses.create(
                model=OPENAI_MODEL or "gpt-5-mini",
                input=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _user_prompt(complaint)},
                ],
                text={"format": {"type": "json_object"}},
            )

        rsp = await asyncio.get_running_loop().run_in_executor(None, _req)
        content = (rsp.output_text or "").strip()

        data = json.loads(content)
        if not isinstance(data, dict) or \
           "services" not in data or "urgency" not in data or "rationale" not in data:
            return False, "LLM returned unexpected format."

        # Normalize a bit
        if isinstance(data.get("services"), list):
            data["services"] = [
                str(s).strip().lower()
                for s in data["services"]
                if str(s).strip()
            ]
        data["urgency"] = str(data.get("urgency", "")).strip().lower()
        data["rationale"] = str(data.get("rationale", "")).strip()

        return True, json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return False, f"LLM error: {e}"


# =========================
# Conversation
# =========================
DETAILS_WAITING = 1


async def details_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /details:
    - DITOLAK kalau origin belum diset
    - Kalau sudah, minta user kirim keluhan (>= 20 kata)
    """
    ok, msg = _origin_is_set(context)
    if not ok:
        await update.message.reply_text(msg)
        return ConversationHandler.END

    origin = context.user_data.get("origin_hospital", {})
    await update.message.reply_text(
        "Please describe your symptoms/condition in detail (minimum 20 words).\n\n"
        f"(Current origin: {origin.get('nama', '-') } / {origin.get('kode', '-')})"
    )
    return DETAILS_WAITING


async def details_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Terima keluhan → cek >=20 kata → panggil LLM → simpan hasil untuk /recommend.
    Tetap re-check origin (jaga-jaga kalau user reset state / bot restart).
    """
    ok, msg = _origin_is_set(context)
    if not ok:
        await update.message.reply_text(msg)
        return ConversationHandler.END

    text = (update.message.text or "").strip()
    wc = _word_count(text)

    if wc < 20:
        await update.message.reply_text(
            f"Your description has {wc} words. "
            "Please use 20 or more words so I can analyze it properly."
        )
        return DETAILS_WAITING

    await update.message.reply_text("Analyzing your details with AI...")

    ok, payload = await _call_llm(text)
    if not ok:
        await update.message.reply_text(
            "Sorry, the AI service had an issue. Please try again.\n"
            f"(debug) {payload}"
        )
        return ConversationHandler.END

    # Simpan untuk /recommend
    context.user_data["complaint"] = text
    context.user_data["llm_result"] = json.loads(payload)

    # Kirim JSON ke user
    await update.message.reply_text(f"LLM Output:\n```\n{payload}\n```", parse_mode="Markdown")
    await update.message.reply_text("Next: use /recommend to get the Best & Alternative hospitals.")
    return ConversationHandler.END


async def details_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Details canceled.")
    return ConversationHandler.END


def get_details_handler() -> ConversationHandler:
    """
    Di bot.py:
        app.add_handler(get_details_handler())
    """
    return ConversationHandler(
        entry_points=[CommandHandler("details", details_entry)],
        states={
            DETAILS_WAITING: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, details_receive),
            ]
        },
        fallbacks=[CommandHandler("cancel", details_cancel)],
        name="details_conversation",
        persistent=False,
    )
