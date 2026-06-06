import json

class PromptBuilder:
    def __init__(self):
        pass

    def _to_builtin(self, obj):
        if isinstance(obj, dict):
            return {self._to_builtin(k): self._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_builtin(v) for v in obj]
        if isinstance(obj, tuple):
            return [self._to_builtin(v) for v in obj]
        if isinstance(obj, (int, float, str, bool)):
            return obj
        try:
            if obj is None:
                return None
            if hasattr(obj, 'tolist'):
                return self._to_builtin(obj.tolist())
        except Exception:
            pass
        return str(obj)

    def build_region_prompt(self, robot_pose, goal, local_patch, uncertain_cells, skill_context=None):
        payload = {
            "task": "classify_uncertain_region",
            "robot_pose": self._to_builtin(robot_pose),
            "goal": self._to_builtin(goal),
            "local_patch": self._to_builtin(local_patch),
            "uncertain_cells": self._to_builtin(uncertain_cells),
            "skill_context": self._to_builtin(skill_context or [])
        }
        instruction = (
            "You are a robot navigation reasoning assistant. "
            "The local patch is a square grid centered on one of the uncertain cells. "
            "Grid values are: 0=free, 1=occupied, -1=uncertain. "
            "Use the map evidence and the robot pose/goal information to classify whether the uncertain cell(s) are most likely free or blocked. "
            "Return only valid JSON with the following fields: label, confidence, reason, recommended_action. "
            "label must be one of likely_free, likely_blocked, uncertain. "
            "confidence must be a number between 0 and 1. "
            "reason must be a short explaining sentence. "
            "recommended_action must be one of plan_through, avoid, increase_cost. "
            "If the uncertain location is bordered by free cells and lies along the path to the goal, prefer likely_free. "
            "If it is adjacent to occupied cells or likely blocks the route, prefer likely_blocked. "
            "If there is not enough evidence, choose uncertain."
        )
        return instruction + "\n" + json.dumps(payload)

    def build_skill_query_prompt(self, situation_context):
        payload = {
            "task": "retrieve_similar_skill",
            "situation_context": situation_context
        }
        instruction = (
            "return only valid json. "
            "skill_id must be a number or null. "
            "score must be a number between 0 and 1."
        )
        return instruction + "\n" + json.dumps(payload)