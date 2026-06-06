import json

class PromptBuilder:
    def __init__(self):
        pass

    def build_region_prompt(self, robot_pose, goal, local_patch, uncertain_cells, skill_context=None):
        payload = {
            "task": "classify_uncertain_region",
            "robot_pose": robot_pose,
            "goal": goal,
            "local_patch": local_patch,
            "uncertain_cells": uncertain_cells,
            "skill_context": skill_context or []
        }
        instruction = (
            "return only valid json. "
            "label must be one of likely_free, likely_blocked, uncertain. "
            "confidence must be a number between 0 and 1. "
            "reason must be a short string. "
            "recommended_action must be one of plan_through, avoid, increase_cost."
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