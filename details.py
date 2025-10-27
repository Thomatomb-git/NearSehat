from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os
import json
import re
import asyncio
from openai import OpenAI
from typing import Final, Tuple
from telegram import Update 
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, ApplicationBuilder,
    ConversationHandler
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

def _word_count(s: str) -> int:
    return len([t for t in re.split(r"\s+", (s or "").strip()) if t])

def _system_prompt() -> str:
    return (
        "You are a concise clinical triage helper. "
        "Given a short free-text complaint in Indonesian or English, "
        "output STRICT JSON with keys: "
        "services (array of strings), urgency (low|medium|high), rationale (1 sentence). "
        "Return JSON ONLY. No extra text. Use English"
    )

def _user_prompt(complaint: str) -> str:
    return (
        "Complaint:\n"
        f"{complaint}\n\n"
        "Map to high-level services (e.g., emergency, pulmonology, cardiology, neurology, "
        "orthopedics, obstetrics_gynecology, pediatrics, cathlab, ct_scan_24h, stroke_unit). "
        "If uncertain, choose the closest high-level services. "
        "Set urgency to high if red-flag symptoms suggest immediate care."
    )

async def _call_llm(complaint: str) -> Tuple[bool, str]:
    """
    Panggil OpenAI Chat Completions (blocking → offloaded via run_in_executor).
    Return (ok, json_text_or_error).
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        def _req():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                # temperature=0.2,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _user_prompt(complaint)},
                ],
                response_format={"type": "json_object"},
                timeout=30,
            )

        rsp = await asyncio.get_running_loop().run_in_executor(None, _req)
        content = (rsp.choices[0].message.content or "").strip()

        # validasi struktur JSON minimal
        data = json.loads(content)
        if not isinstance(data, dict) or \
           "services" not in data or "urgency" not in data or "rationale" not in data:
            return False, "LLM returned unexpected format."

        # normalisasi ringan
        if isinstance(data.get("services"), list):
            data["services"] = [str(s).strip().lower() for s in data["services"] if str(s).strip()]
        data["urgency"] = str(data.get("urgency", "")).strip().lower()

        # re-dump agar rapi
        return True, json.dumps(data, ensure_ascii=False)

    except Exception as e:
        return False, f"LLM error: {e}"


DETAILS_WAITING = 1

async def details_entry(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /details → minta user kirimkan deskripsi keluhan (free text).
    """
    await update.message.reply_text(
        "Please describe your symptoms/condition in detail (minimum 20 words)."
    )
    return DETAILS_WAITING

async def details_receive(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Terima keluhan → cek >=20 kata → panggil LLM → balikan JSON & simpan.
    """
    text = (update.message.text or "").strip()
    wc = _word_count(text)

    if wc < 20:
        await update.message.reply_text(
            f"Your description has {wc} words. "
            f"Please use 20 or more words so I can analyze it properly."
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

    # Kirim JSON ke user (sesuai spesifikasi)
    await update.message.reply_text(f"LLM Output:\n```\n{payload}\n```", parse_mode="Markdown")
    await update.message.reply_text("Next: use /recommend to get the Best & Alternative hospitals.")
    return ConversationHandler.END

async def details_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Details canceled.")
    return ConversationHandler.END

def get_details_handler() -> ConversationHandler:
    """
    Daftarkan ini ke Application kamu:
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