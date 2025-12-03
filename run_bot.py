import os
import dotenv


# apply load .env file
dotenv.load_dotenv(".env")

from bot.bot import main

if __name__ == "__main__":
    main()