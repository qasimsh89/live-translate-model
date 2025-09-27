import os
from dotenv import load_dotenv, find_dotenv, dotenv_values

print("Looking for .env ...")
env_path = find_dotenv()
print("find_dotenv():", env_path if env_path else "NOT FOUND")

load_dotenv(env_path, override=True)

print("AAI_API_KEY from os.getenv:", os.getenv("AAI_API_KEY"))
if env_path:
    print("All keys in .env:", list(dotenv_values(env_path).keys()))
