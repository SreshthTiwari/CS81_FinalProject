import numpy as np

class Replanner:
    def __init__(self, prompt_builder, response_parser, llm_client, context_extractor):
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.llm_client = llm_client
        self.context_extractor = context_extractor
        self.last_prompt = None
        self.last_response = None
        self.last_patch = None

    def get_debug_info(self):
        return {
            'prompt': self.last_prompt,
            'response': self.last_response,
            'patch': self.last_patch,
        }

    def get_path_neighborhood(self, grid, path, radius=2):
        interesting = []
        height, width = grid.shape

        for cell in path:
            cx, cy = cell
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        if grid[y, x] == -1 or grid[y, x] == 1:
                            interesting.append((x, y))

        return list(set(interesting))

    def classify_region(self, grid, robot_pose, goal, target_cell, skill_context=None, situation_type=None):
        patch = self.context_extractor.extract_patch(grid, target_cell)
        prompt = self.prompt_builder.build_region_prompt(
            robot_pose=robot_pose,
            goal=goal,
            local_patch=patch,
            observed_cells=[target_cell],
            skill_context=skill_context,
            situation_type=situation_type
        )
        response_text = self.llm_client.query(prompt)
        self.last_prompt = prompt
        self.last_response = response_text
        self.last_patch = patch
        result = self.response_parser.parse_region_response(response_text)
        return result, patch

    def apply_decision_to_grid(self, grid, target_cells, decision):
        modified = grid.copy()
        action = decision.get("recommended_action", "")

        if decision["label"] == "likely_blocked" and action in ("replan_immediately", "avoid"):
            for x, y in target_cells:
                modified[y, x] = 1

        elif decision["label"] == "likely_free":
            for x, y in target_cells:
                if modified[y, x] == -1:
                    modified[y, x] = 0

        return modified

    def replan(self, grid, start, goal, path, robot_pose, skill_context=None, original_path=None, situation_type=None):
        path_to_scan = path if path else original_path or []
        interesting_cells = self.get_path_neighborhood(grid, path_to_scan)
        if not interesting_cells:
            return grid, None

        target_cell = interesting_cells[0]
        decision, patch = self.classify_region(
            grid=grid,
            robot_pose=robot_pose,
            goal=goal,
            target_cell=target_cell,
            skill_context=skill_context,
            situation_type=situation_type
        )

        modified_grid = self.apply_decision_to_grid(grid, [target_cell], decision)
        return modified_grid, decision
    def classify_temporal_obstacle(self, grids, robot_pose, goal, target_cell, skill_context=None, situation_type=None):
        """Analyze obstacle across multiple timesteps."""
        timestep_patches = []
        for grid in grids:
            patch = self.context_extractor.extract_patch(grid, target_cell)
            timestep_patches.append(patch)

        prompt = self.prompt_builder.build_temporal_obstacle_prompt(
            robot_pose=robot_pose,
            goal=goal,
            timestep_patches=timestep_patches,
            observed_cells=[target_cell],
            num_timesteps=len(grids),
            skill_context=skill_context,
            situation_type=situation_type
        )
        response_text = self.llm_client.query(prompt)
        self.last_prompt = prompt
        self.last_response = response_text
        self.last_patch = timestep_patches
        result = self.response_parser.parse_temporal_obstacle_response(response_text)
        return result

    def replan_temporal(self, grids, start, goal, original_path, robot_pose, skill_context=None, situation_type=None, target_cell=None):
        """Replan using temporal obstacle analysis."""
        if not grids or not original_path:
            return grids[0].copy() if grids else None, None

        if target_cell is None:
            interesting_cells = self.get_path_neighborhood(grids[0], original_path)
            if not interesting_cells:
                return grids[0].copy(), None

            original_path_set = set(original_path)
            blocked_path_cells = [cell for cell in interesting_cells
                                  if cell in original_path_set and grids[0][cell[1], cell[0]] == 1]
            if not blocked_path_cells:
                return grids[0].copy(), {
                    "label": "likely_free",
                    "movement_pattern": "vanishing",
                    "confidence": 0.85,
                    "reason": "No blockage remains on the current planned path.",
                    "recommended_action": "plan_through"
                }
            target_cell = blocked_path_cells[0]

        # Heuristic: if uncertain cells (-1) lie on the original path and persist across
        # the majority of timesteps, treat them as a likely static blockage and replan.
        try:
            original_path_set = set(original_path)
            persistent_uncertain = []
            for cell in original_path_set:
                x, y = cell
                count_uncertain = 0
                for g in grids:
                    if g[y, x] == -1:
                        count_uncertain += 1
                if count_uncertain >= (len(grids) + 1) // 2:
                    persistent_uncertain.append(cell)
            if persistent_uncertain:
                # mark these cells as blocked on a copy and recommend replanning
                modified = grids[0].copy()
                for x, y in persistent_uncertain:
                    modified[y, x] = 1
                decision = {
                    "label": "likely_blocked",
                    "movement_pattern": "static",
                    "confidence": 0.9,
                    "reason": "Uncertain observations persist on the planned path across timesteps",
                    "recommended_action": "replan_immediately"
                }
                return modified, decision
        except Exception:
            # Fall back to classifier below on any failure
            pass

        decision = self.classify_temporal_obstacle(
            grids=grids,
            robot_pose=robot_pose,
            goal=goal,
            target_cell=target_cell,
            skill_context=skill_context,
            situation_type=situation_type
        )

        modified_grid = self.apply_decision_to_grid(grids[0], [target_cell], decision)
        return modified_grid, decision
