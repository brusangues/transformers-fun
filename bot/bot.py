"""
start - inicializar o ambiente
reset - reiniciar o ambiente
echo - retorna a mensagem do usuário
help - ajuda
load - carrega um modelo de linguagem
rag - carrega modelo de embedding e índice de busca semântica
filter - configura um filtro para o índice de busca semântica
"""

# pip install python-telegram-bot
import os
import traceback
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import torch
import json

from .models_api import (
    load_model,
    query_model,
    models_text,
)
from .agent import answer_with_local_model
from .models_local import load_local_model

MODEL_API = None
MODEL_LOCAL = None
FILTER = {}
MESSAGE_LIMIT = 4095


# Commands
def load_models(
    model_api_alias = "gemini-2.5-flash",
    model_local_name = "v22",
):
    global MODEL_API, MODEL_LOCAL
    del MODEL_API, MODEL_LOCAL
    torch.cuda.empty_cache()
    print(f"load_models: {model_api_alias=} {model_local_name=}")
    MODEL_API, model_name, max_new_tokens = load_model(model_api_alias)
    messages = [
        f"Modelo api carregado: {model_name}\nCom máximo de tokens: {max_new_tokens}",
    ]
    
    if model_local_name is None:
        MODEL_LOCAL = None
    else:
        MODEL_LOCAL = load_local_model(model_local_name)
        messages.append("Modelo local carregado.")
    for m in messages:
        print(m)
    return messages


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nstart_command")
    messages = load_models()
    for m in messages:
        await update.message.reply_text(m)
    await update.message.reply_text("Oi, eu sou um bot. Ambiente inicializado!")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global MODEL_API, MODEL_LOCAL
    print("\nreset_command")
    del MODEL_API, MODEL_LOCAL
    torch.cuda.empty_cache()
    MODEL_API = None
    MODEL_LOCAL = None
    await update.message.reply_text("Ambiente reinicializado!")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nhelp_command")
    await update.message.reply_text(
        f"Modelos de linguagem via api disponíveis:\n{models_text}\n"
        f"Comandos disponíveis: start, echo, load, rag ,filter, help"
    )


async def echo_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\necho_command")
    await update.message.reply_text(update.message.text)


async def load_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nload_command")
    args = update.message.text.replace("/load", "").strip().split()
    print(f"{args=}")
    messages = load_models(*args)
    for m in messages:
        await update.message.reply_text(m)
    await update.message.reply_text("Modelos carregados!")


# Messages
async def handle_response(update: Update, text: str) -> str:
    global MODEL_API, MODEL_LOCAL
    if "oi" == text.lower():
        return "Oi de novo, eu sou um bot!"
    elif MODEL_API is not None and MODEL_LOCAL is not None:
        response = answer_with_local_model(MODEL_API, MODEL_LOCAL, text)
        print(f"Response length: {len(response)}/{MESSAGE_LIMIT}")
        response = response[:MESSAGE_LIMIT]
        return response
    elif MODEL_API is not None:
        response = query_model(MODEL_API, text)[0]
        print(f"Response length: {len(response)}/{MESSAGE_LIMIT}")
        response = response[:MESSAGE_LIMIT]
        return response
    return (
        "Nenhum modelo está carregado. "
        "Use o comando /load para carregar um modelo de linguagem. "
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nhandle_message")
    print(f"{update=}\n{context=}")

    try:
        message_type = update.message.chat.type
        text = update.message.text
        print(f"User ({update.message.chat.id}) in {message_type}: {text}")

        response = await handle_response(update, text)
        print(f"Response length: {len(response)}/{MESSAGE_LIMIT}")
        response = response[:MESSAGE_LIMIT]

        print(f"Bot: {response}")
        await update.message.reply_text(response)

    except Exception as e:
        print(e)
        traceback.print_exc()
        raise e


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("\nerror")
    print(f"{update=}\n{context.error=}")
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Desculpe, ocorreu um erro!",
    )


def main():
    print("Starting bot...")
    app = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()

    # Commands
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("echo", echo_command))
    app.add_handler(CommandHandler("load", load_command))
    # app.add_handler(CommandHandler("rag", rag_command))
    # app.add_handler(CommandHandler("filter", filter_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors
    app.add_error_handler(error)

    load_models()

    print("Polling...")
    app.run_polling(poll_interval=3)



# Main
if __name__ == "__main__":
    main()
