from dotenv import load_dotenv
load_dotenv()
import os
import json
import re
import asyncio
import openai
from typing import Final
from telegram import Update 
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, ApplicationBuilder,
    ConversationHandler
)
from details import get_details_handler
from setorigin import get_setorigin_handler
from recommend import get_recommend_handler

BOT_TOKEN = os.getenv("BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to NearSehat!\n\n"
        "I can help you find the best hospital for your needs.\n"
        "Type /help to see available commands."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this message\n"
        "/setorigin - Set your current hospital or location\n"
        "/details - Describe your health condition\n"
        "/recommend - Get hospital recommendations\n"
    )

async def setorigin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Later updated")

async def details(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Later updated")

async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Later updated")


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(get_setorigin_handler())
    app.add_handler(get_details_handler())
    app.add_handler(get_recommend_handler())
    
    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()