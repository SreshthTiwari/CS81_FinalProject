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

    def build_region_prompt(self, robot_pose, goal, local_patch, observed_cells, skill_context=None, situation_type=None):
        payload = {
            "task": "classify_navigation_obstacle",
            "situation_type": self._to_builtin(situation_type),
            "robot_pose": self._to_builtin(robot_pose),
            "goal": self._to_builtin(goal),
            "local_patch": self._to_builtin(local_patch),
            "observed_cells": self._to_builtin(observed_cells),
            "skill_context": self._to_builtin(skill_context or [])
        }
        instruction = (
            "You are a robot navigation reasoning assistant. "
            "The local patch is a square grid centered on the observed cell. "
            "Grid values are: 0=free, 1=occupied/blocked, -1=uncertain. "
            "This query includes a `situation_type` which can be one of: 'uncertain' (noise as -1), "
            "or 'new_blockage' (a newly observed blockage marked as 1). "
            "When situation_type is 'uncertain', return one of these recommended_action values: "
            "keep_moving, replan, or wait_and_reinspect. Prefer keep_moving when the uncertain cells move or are transient. "
            "When situation_type is 'new_blockage', return one of: wait_and_reinspect or replan. Prefer replan when blockage is static and blocks the path. "
            "Return only valid JSON with the following fields: label, confidence, reason, recommended_action. "
            "label must be one of likely_free, likely_blocked, uncertain. "
            "confidence must be a number between 0 and 1. "
            "reason must be a short explanatory sentence. "
            "recommended_action must follow the rules above for the provided situation_type."
        )
        return instruction + "\n" + json.dumps(payload)

    def build_temporal_obstacle_prompt(self, robot_pose, goal, timestep_patches, observed_cells, num_timesteps=3, skill_context=None, situation_type=None):
        """Build a prompt for temporal obstacle analysis (multi-timestep data).
        
        Args:
            robot_pose: dict with robot position
            goal: dict with goal position
            timestep_patches: list of patches for each timestep [t0_patch, t1_patch, t2_patch]
            observed_cells: list of cell coordinates being observed
            num_timesteps: number of timesteps
            skill_context: optional skill context
        """
        payload = {
            "task": "classify_temporal_obstacle",
            "situation_type": self._to_builtin(situation_type),
            "robot_pose": self._to_builtin(robot_pose),
            "goal": self._to_builtin(goal),
            "num_timesteps": num_timesteps,
            "timestep_patches": [
                {
                    "timestep": idx,
                    "patch": self._to_builtin(patch)
                }
                for idx, patch in enumerate(timestep_patches)
            ],
            "observed_cells": self._to_builtin(observed_cells),
            "skill_context": self._to_builtin(skill_context or [])
        }
        instruction = (
            "You are a robot navigation reasoning assistant analyzing temporal obstacle data. "
            "The robot has taken observations over multiple timesteps (t=0, t=1, ...). "
            "Each timestep_patches entry contains a timestep index and a 7x7 patch centered on the observed cell: 0=free, 1=occupied/blocked, -1=uncertain. "
            "This query includes an explicit `situation_type` which can be 'uncertain' (noise as -1) or 'new_blockage' (1s). "
            "If the uncertain (-1) cells are the ones that move across timesteps, treat this as 'uncertain' and prefer recommended_action keep_moving. "
            "If 1s move across timesteps, treat this as a moving blockage and the best action is wait_and_reinspect. "
            "If 1s persist in approximately the same locations across timesteps, treat this as a permanent blockage and recommend replan. "
            "Return only valid JSON with fields: label, movement_pattern, confidence, reason, recommended_action. "
            "movement_pattern must be one of static, moving, vanishing, or unknown. "
            "For situation_type 'uncertain', return one of: keep_moving, replan, wait_and_reinspect. "
            "For situation_type 'new_blockage', return one of: wait_and_reinspect, replan. "
            "Use wait_and_reinspect only when a new blockage is moving; if 1s remain persistent, choose replan. "
            "Provide a short reason explaining the movement evidence and confidence."
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