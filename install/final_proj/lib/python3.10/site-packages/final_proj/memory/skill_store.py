import json
from pathlib import Path

class SkillStore:
    def __init__(self, store_path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if self.store_path.exists():
            with open(self.store_path, "r") as f:
                self.skills = json.load(f)
        else:
            self.skills = []

    def add_skill(self, start_context, end_context, path_pattern, outcome):
        skill = {
            "id": len(self.skills),
            "start_context": start_context,
            "end_context": end_context,
            "path_pattern": path_pattern,
            "outcome": outcome
        }
        self.skills.append(skill)
        self.save()
        return skill["id"]

    def save(self):
        with open(self.store_path, "w") as f:
            json.dump(self.skills, f, indent=2)

    def get_all_skills(self):
        return self.skills

    def find_skill_by_id(self, skill_id):
        for skill in self.skills:
            if skill["id"] == skill_id:
                return skill
        return None