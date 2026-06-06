import numpy as np

class Replanner:
    def __init__(self, prompt_builder, response_parser, llm_client, context_extractor):
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.llm_client = llm_client
        self.context_extractor = context_extractor

    def get_path_neighborhood(self, grid, path, radius=2):
        uncertain = []
        height, width = grid.shape

        for cell in path:
            cx, cy = cell
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < width and 0 <= y < height:
                        if grid[y, x] == -1:
                            uncertain.append((x, y))

        return list(set(uncertain))

    def classify_region(self, grid, robot_pose, goal, target_cell, skill_context=None):
        patch = self.context_extractor.extract_patch(grid, target_cell)
        prompt = self.prompt_builder.build_region_prompt(
            robot_pose=robot_pose,
            goal=goal,
            local_patch=patch,
            uncertain_cells=[target_cell],
            skill_context=skill_context
        )
        response_text = self.llm_client.query(prompt)
        result = self.response_parser.parse_region_response(response_text)
        return result, patch

    def apply_decision_to_grid(self, grid, target_cells, decision):
        modified = grid.copy()

        if decision["label"] == "likely_blocked":
            for x, y in target_cells:
                modified[y, x] = 1

        elif decision["label"] == "likely_free":
            for x, y in target_cells:
                if modified[y, x] == -1:
                    modified[y, x] = 0

        else:
            for x, y in target_cells:
                if modified[y, x] == -1:
                    modified[y, x] = -1

        return modified

    def replan(self, grid, start, goal, path, robot_pose, skill_context=None):
        uncertain_cells = self.get_path_neighborhood(grid, path)
        if not uncertain_cells:
            return grid, None

        target_cell = uncertain_cells[0]
        decision, patch = self.classify_region(
            grid=grid,
            robot_pose=robot_pose,
            goal=goal,
            target_cell=target_cell,
            skill_context=skill_context
        )

        modified_grid = self.apply_decision_to_grid(grid, [target_cell], decision)
        return modified_grid, decision