import json
import os

try:
    from groq import Groq
except ImportError:
    Groq = None

class LLMClient:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = None
        self.is_stub = False

        if Groq is None:
            print("[WARN] groq is not installed; using local LLM stub fallback.")
            self.is_stub = True
            return

        api_key = os.environ.get("GROQ_API_KEY")
        if api_key is None:
            print("[WARN] GROQ_API_KEY is not set; using local LLM stub fallback.")
            self.is_stub = True
            return

        self.client = Groq(api_key=api_key)

    def query(self, prompt):
        if self.is_stub or self.client is None:
            return self.stub_response(prompt)

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

    def stub_response(self, prompt):
        # Return a valid JSON fallback so the replanner can still run
        try:
            payload = prompt.split("\n", 1)[1]
            data = json.loads(payload)
            local_patch = data.get("local_patch", [])
            uncertain_cells = data.get("uncertain_cells", [])
        except Exception:
            local_patch = []
            uncertain_cells = []

        free_count = 0
        occupied_count = 0
        uncertain_count = 0

        for row in local_patch:
            for cell in row:
                if cell == 0:
                    free_count += 1
                elif cell == 1:
                    occupied_count += 1
                elif cell == -1:
                    uncertain_count += 1

        if not uncertain_cells:
            label = "uncertain"
            confidence = 0.0
            reason = "no uncertain cells provided"
            recommended_action = "uncertain"
        elif occupied_count > free_count:
            label = "likely_blocked"
            confidence = min(0.9, 0.4 + 0.1 * (occupied_count - free_count))
            reason = "uncertain region is surrounded by obstacles"
            recommended_action = "avoid"
        elif free_count > occupied_count:
            label = "likely_free"
            confidence = min(0.9, 0.4 + 0.1 * (free_count - occupied_count))
            reason = "uncertain region is surrounded by free space"
            recommended_action = "plan_through"
        else:
            label = "uncertain"
            confidence = 0.45
            reason = "insufficient evidence in the local patch"
            recommended_action = "increase_cost"

        return json.dumps({
            "label": label,
            "confidence": float(confidence),
            "reason": reason,
            "recommended_action": recommended_action
        })