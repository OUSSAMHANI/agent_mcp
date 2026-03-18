from langchain_google_genai import ChatGoogleGenerativeAI
import os


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Initializes the Gemini 2.0 Flash client.
    Requires GOOGLE_API_KEY to be set in your .env file.
    Free tier: https://aistudio.google.com/app/apikey
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        streaming=True,
        temperature=0.1,
    )