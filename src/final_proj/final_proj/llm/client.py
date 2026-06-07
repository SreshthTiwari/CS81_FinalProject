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
            try:
                payload = prompt.split("\n", 1)[1]
                data = json.loads(payload)
                task = data.get("task", "")
            except Exception:
                task = ""

            if task == "classify_temporal_obstacle":
                return self.temporal_stub_response(prompt)
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
            observed_cells = data.get("observed_cells", [])
        except Exception:
            local_patch = []
            observed_cells = []

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

        if not observed_cells:
            label = "uncertain"
            confidence = 0.0
            reason = "no observed cells provided"
            recommended_action = "increase_cost"
        elif uncertain_count > 0:
            if occupied_count > free_count:
                label = "likely_blocked"
                confidence = min(0.9, 0.4 + 0.1 * (occupied_count - free_count))
                reason = "uncertain region is surrounded by obstacles"
                recommended_action = "avoid"
            else:
                label = "likely_free"
                confidence = min(0.9, 0.4 + 0.1 * (free_count - occupied_count))
                reason = "uncertain region is surrounded by free space"
                recommended_action = "plan_through"
        elif occupied_count > 0:
            label = "likely_blocked"
            confidence = min(0.9, 0.5 + 0.05 * (occupied_count - free_count))
            if free_count >= occupied_count + 3:
                reason = "blocked cell appears isolated and temporary"
                recommended_action = "wait"
            elif free_count >= occupied_count:
                reason = "blocked cell is present but local context still allows inspection"
                recommended_action = "inspect"
            else:
                reason = "blocked region is surrounded by obstacles and likely requires replanning"
                recommended_action = "replan_immediately"
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

    def temporal_stub_response(self, prompt):
        """Analyze temporal obstacle data from multiple timesteps."""
        try:
            payload = prompt.split("\n", 1)[1]
            data = json.loads(payload)
            timestep_patches = data.get("timestep_patches", [])
        except Exception:
            timestep_patches = []

        if not timestep_patches or len(timestep_patches) < 2:
            return json.dumps({
                "label": "uncertain",
                "movement_pattern": "unknown",
                "confidence": 0.0,
                "reason": "insufficient timestep data",
                "recommended_action": "increase_cost"
            })

        # Analyze each timestep for obstacle positions
        timestep_obstacle_positions = []
        for patch_entry in timestep_patches:
            patch = patch_entry.get("patch") if isinstance(patch_entry, dict) else patch_entry
            positions = set()
            for y, row in enumerate(patch):
                for x, cell in enumerate(row):
                    if cell == 1:
                        positions.add((x, y))
            timestep_obstacle_positions.append(positions)

        is_vanishing = bool(timestep_obstacle_positions[0]) and not bool(timestep_obstacle_positions[-1])
        is_static = len(timestep_obstacle_positions) > 0 and all(
            positions == timestep_obstacle_positions[0] for positions in timestep_obstacle_positions
        ) and bool(timestep_obstacle_positions[0])
        is_moving = not is_static and any(
            timestep_obstacle_positions[i] != timestep_obstacle_positions[i + 1]
            for i in range(len(timestep_obstacle_positions) - 1)
        ) and bool(timestep_obstacle_positions[0])

        total_occupied = sum(len(positions) for positions in timestep_obstacle_positions)

        if is_moving:
            label = "likely_blocked"
            movement_pattern = "moving"
            confidence = min(0.85, 0.5 + 0.1 * len(timestep_obstacle_positions))
            reason = "obstacle pattern shifts across timesteps; temporary moving blockage detected"
            recommended_action = "wait_and_reinspect"
        elif is_vanishing:
            label = "likely_blocked"
            movement_pattern = "vanishing"
            confidence = 0.8
            reason = "obstacle disappears in later timesteps; likely passing through"
            recommended_action = "wait"
        elif is_static:
            label = "likely_blocked"
            movement_pattern = "static"
            confidence = min(0.9, 0.6 + 0.1 * len(timestep_obstacle_positions))
            reason = "obstacle remains in the same place across timesteps; likely permanent blockage"
            recommended_action = "replan_immediately"
        elif total_occupied == 0:
            label = "likely_free"
            movement_pattern = "vanishing"
            confidence = 0.6
            reason = "no obstacle remains over the observed timesteps"
            recommended_action = "plan_through"
        else:
            label = "uncertain"
            movement_pattern = "unknown"
            confidence = 0.5
            reason = "temporal evidence is inconclusive for obstacle classification"
            recommended_action = "increase_cost"

        return json.dumps({
            "label": label,
            "movement_pattern": movement_pattern,
            "confidence": float(confidence),
            "reason": reason,
            "recommended_action": recommended_action
        })
