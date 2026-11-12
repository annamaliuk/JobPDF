import os

from jobpdf.extraction.llm import load_env_file

load_env_file()
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
try:
    reply = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL") or "gemini-3.1-flash-lite", contents="Say ok"
    )
    print("ok:", reply.text)
except Exception as exc:
    print(type(exc).__name__, getattr(exc, "code", None), getattr(exc, "status", None))
    print(str(getattr(exc, "message", ""))[:300])