import json

class ResponseParser:
    def __init__(self):
        pass

    def parse_region_response(self, response_text):
        data = json.loads(response_text.strip())
        label = data.get("label", "uncertain")
        confidence = data.get("confidence", 0.0)
        if confidence is None:
            confidence = 0.0
        reason = data.get("reason", "")
        recommended_action = data.get("recommended_action", "uncertain")
        return {
            "label": label,
            "confidence": float(confidence),
            "reason": reason,
            "recommended_action": recommended_action
        }

    def parse_skill_response(self, response_text):
        data = json.loads(response_text.strip())
        skill_id = data.get("skill_id", None)
        score = data.get("score", 0.0)
        if score is None:
            score = 0.0
        return {
            "skill_id": skill_id,
            "score": float(score)
        }