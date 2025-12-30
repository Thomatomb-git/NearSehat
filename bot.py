from __future__ import annotations

# NearSehat Bot - single guided flow + /reason + inline buttons (Reason + Home)
# Semua output ke user harus dalam bahasa Inggris.
# Comment boleh pakai bahasa Indonesia.

from dotenv import load_dotenv
load_dotenv()

import os
import logging
import re
from typing import Any, Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from setorigin import (
    ORIGIN_QUERY,
    CALLBACK_PREFIX_ORIGIN,
    search_hospitals,
    build_origin_keyboard,
    get_hospital_by_kode,
)
from details import (
    COMPLAINT_INPUT,
    generate_recommendation_criteria,
)
from recommend import (
    get_allowed_facility_tokens,
    build_recommendation_payload,
    render_main_message,
    render_reason_message,
)

# =========================
# Logging
# =========================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("nearsehat")

BOT_TOKEN = os.getenv("BOT_TOKEN")

HOME_CALLBACK = "HOME"
REASON_CALLBACK = "REASON"


def _word_count(s: str) -> int:
    return len([t for t in re.split(r"\s+", (s or "").strip()) if t])


def _home_text() -> str:
    return (
        "🏥 <b>NearSehat</b>\n\n"
        "Type /start to begin.\n"
        "Type /help for commands."
    )


def _action_keyboard(include_reason: bool = True) -> InlineKeyboardMarkup:
    # Satu row, sejajar: Reason | Home
    row = []
    if include_reason:
        row.append(InlineKeyboardButton("🧠 Reason", callback_data=REASON_CALLBACK))
    row.append(InlineKeyboardButton("🏠 Home", callback_data=HOME_CALLBACK))
    return InlineKeyboardMarkup([row])


# =========================
# Home / Help
# =========================
async def home_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    await update.effective_message.reply_text(_home_text(), parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🆘 <b>Help</b>\n\n"
        "/start — Start the guided flow (choose origin → describe symptoms → get recommendation)\n"
        "/reason — Explain why the hospitals were chosen (after a recommendation)\n"
        "/home — Go back to the home page and reset\n"
        "/cancel — Cancel the current flow and reset\n"
        "/help — Show this message"
    )
    await update.message.reply_text(text, parse_mode="HTML")


# =========================
# /reason (manual command)
# =========================
async def reason_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    payload = context.user_data.get("last_recommendation")
    if not payload:
        await update.message.reply_text(
            "No recent recommendation found.\n\nType /start to begin."
        )
        return

    try:
        text = render_reason_message(payload)
    except Exception as e:
        logger.exception("Failed to render reason: %s", e)
        await update.message.reply_text(
            "⚠️ I couldn't show the reasoning due to an internal error."
        )
        return

    await update.message.reply_text(text, parse_mode="HTML", reply_markup=_action_keyboard(include_reason=False))


# =========================
# Callbacks: Home / Reason
# =========================
async def home_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    # Clear all user data (termasuk last_recommendation)
    context.user_data.clear()

    # Kirim home sebagai message baru (biar rekomendasi tidak hilang)
    await query.message.reply_text(_home_text(), parse_mode="HTML")

    # Hapus tombol dari message lama (opsional, biar clean)
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass


async def reason_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    payload = context.user_data.get("last_recommendation")
    if not payload:
        await query.message.reply_text(
            "No recent recommendation found.\n\nType /start to begin."
        )
        return

    try:
        text = render_reason_message(payload)
    except Exception as e:
        logger.exception("Failed to render reason (callback): %s", e)
        await query.message.reply_text(
            "⚠️ I couldn't show the reasoning due to an internal error."
        )
        return

    # Kirim reasoning sebagai message baru + tombol Home (Reason nggak perlu di reason message)
    await query.message.reply_text(text, parse_mode="HTML", reply_markup=_action_keyboard(include_reason=False))


# =========================
# Cancel (global)
# =========================
async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.effective_message.reply_text(_home_text(), parse_mode="HTML")
    return ConversationHandler.END


# =========================
# /start entry (guided flow)
# =========================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()

    try:
        _ = get_allowed_facility_tokens()
    except Exception as e:
        logger.exception("Failed to load facility tokens: %s", e)
        await update.message.reply_text(
            "⚠️ Data files could not be loaded. Please try again later."
        )
        return ConversationHandler.END

    await update.message.reply_text(
        "Welcome to NearSehat.\n\n"
        "Step 1/3 — <b>Choose your current hospital (origin).</b>\n"
        "Please type the hospital name or code:",
        parse_mode="HTML",
    )
    return ORIGIN_QUERY


# =========================
# State: ORIGIN_QUERY
# =========================
async def origin_query_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    q = (update.message.text or "").strip()
    if not q:
        await update.message.reply_text(
            "Please type a hospital name or code (or /cancel)."
        )
        return ORIGIN_QUERY

    results = search_hospitals(q, limit=10)
    if not results:
        await update.message.reply_text(
            "No matches found. Please try another keyword (or /cancel)."
        )
        return ORIGIN_QUERY

    kb = build_origin_keyboard(results)
    await update.message.reply_text(
        "Select your hospital:",
        reply_markup=kb,
    )
    return ORIGIN_QUERY


async def origin_select_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if not data.startswith(CALLBACK_PREFIX_ORIGIN):
        return ORIGIN_QUERY

    kode = data[len(CALLBACK_PREFIX_ORIGIN):].strip()
    hosp = get_hospital_by_kode(kode)
    if not hosp:
        await query.edit_message_text(
            "Sorry, that hospital is no longer available. Please search again."
        )
        return ORIGIN_QUERY

    context.user_data["origin_hospital"] = {
        "kode": hosp.kode,
        "nama": hosp.nama,
        "jenis": hosp.jenis,
        "kelas": hosp.kelas,
        "alamat": hosp.alamat,
    }

    await query.edit_message_text(
        f"✅ Origin set: <b>{hosp.nama}</b> ({hosp.kode})\n\n"
        "Step 2/3 — <b>Describe your symptoms / needs.</b>\n"
        "Please write at least 20 words:",
        parse_mode="HTML",
    )
    return COMPLAINT_INPUT


# =========================
# State: COMPLAINT_INPUT
# =========================
async def complaint_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    complaint = (update.message.text or "").strip()
    if not complaint:
        await update.message.reply_text("Please describe your symptoms (or /cancel).")
        return COMPLAINT_INPUT

    if _word_count(complaint) < 20:
        await update.message.reply_text(
            "Please write a bit more detail (at least 20 words), so I can triage accurately.\n"
            "You can also type /cancel."
        )
        return COMPLAINT_INPUT

    status_msg = await update.message.reply_text("🤖 Analyzing your symptoms with AI...")

    allowed_tokens = get_allowed_facility_tokens()
    ok, crit_or_err = await generate_recommendation_criteria(
        complaint=complaint,
        allowed_tokens=allowed_tokens,
    )
    if not ok:
        try:
            await status_msg.edit_text("⚠️ Analysis failed.")
        except Exception:
            pass

        await update.message.reply_text(
            "⚠️ I couldn't analyze your message right now.\n"
            f"Reason: {crit_or_err}\n\n"
            "Please try sending your symptoms again, or type /cancel."
        )
        return COMPLAINT_INPUT

    try:
        await status_msg.edit_text("✅ Analysis complete. Generating recommendations...")
    except Exception:
        pass

    criteria: Dict[str, Any] = crit_or_err  # type: ignore
    origin = context.user_data.get("origin_hospital")
    if not origin:
        await update.message.reply_text(
            "⚠️ Origin hospital is not set. Please type /start to begin again."
        )
        context.user_data.clear()
        return ConversationHandler.END

    try:
        payload = build_recommendation_payload(
            origin_kode=str(origin["kode"]),
            origin_name=str(origin["nama"]),
            criteria=criteria,
        )
        main_text = render_main_message(payload)
    except Exception as e:
        logger.exception("Recommendation failed: %s", e)
        await update.message.reply_text(
            "⚠️ I couldn't generate a recommendation due to a data error.\n"
            "Please try again later or contact the bot admin."
        )
        context.user_data.clear()
        return ConversationHandler.END

    context.user_data["last_recommendation"] = payload

    await update.message.reply_text(
        main_text,
        parse_mode="HTML",
        reply_markup=_action_keyboard(include_reason=True),  # <-- Reason sejajar Home
    )

    # Clear flow state but keep last_recommendation
    last = context.user_data.get("last_recommendation")
    context.user_data.clear()
    context.user_data["last_recommendation"] = last

    return ConversationHandler.END


def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is not set in environment variables.")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            ORIGIN_QUERY: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, origin_query_message),
                CallbackQueryHandler(origin_select_callback, pattern=f"^{CALLBACK_PREFIX_ORIGIN}"),
            ],
            COMPLAINT_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, complaint_message),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
        allow_reentry=True,
        persistent=False,
        name="nearsehat_main_flow",
    )

    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("home", home_command))
    app.add_handler(CommandHandler("reason", reason_command))
    app.add_handler(CommandHandler("cancel", cancel_command))

    # callback buttons
    app.add_handler(CallbackQueryHandler(home_callback, pattern=f"^{HOME_CALLBACK}$"))
    app.add_handler(CallbackQueryHandler(reason_callback, pattern=f"^{REASON_CALLBACK}$"))

    app.add_handler(conv)

    logger.info("NearSehat bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
