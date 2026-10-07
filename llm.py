import os
from dotenv import load_dotenv
from google import genai

# Load variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Send a test query
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="What is semantic caching? Explain in one sentence."
)
print(response.text)