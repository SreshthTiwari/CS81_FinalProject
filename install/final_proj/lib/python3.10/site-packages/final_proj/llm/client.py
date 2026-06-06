import os
from groq import Groq

class LLMClient:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("GROQ_API_KEY is not set")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def query(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "you are a navigation reasoning assistant. "
                        "return only valid json. "
                        "do not include markdown, code fences, or extra text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0,
            max_tokens=256
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()