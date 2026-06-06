import math

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def compare_patches(patch_a, patch_b):
    if not patch_a or not patch_b:
        return 0.0
    if len(patch_a) != len(patch_b):
        return 0.0
    if len(patch_a) == 0:
        return 0.0

    same = 0
    total = 0
    for row_a, row_b in zip(patch_a, patch_b):
        if len(row_a) != len(row_b):
            return 0.0
        for va, vb in zip(row_a, row_b):
            total += 1
            if va == vb:
                same += 1
    if total == 0:
        return 0.0
    return same / total

def compare_direction(dir_a, dir_b):
    if dir_a == dir_b:
        return 1.0
    return 0.0

def compare_turn_patterns(turns_a, turns_b):
    if not turns_a or not turns_b:
        return 0.0
    overlap = min(len(turns_a), len(turns_b))
    if overlap == 0:
        return 0.0
    same = 0
    for i in range(overlap):
        if turns_a[i] == turns_b[i]:
            same += 1
    return same / overlap

def compare_path_length(len_a, len_b):
    denom = max(len_a, len_b, 1)
    return 1.0 - abs(len_a - len_b) / denom

def similarity_score(current_context, stored_context):
    patch_score = compare_patches(
        current_context.get("start_patch", []),
        stored_context.get("start_patch", [])
    )
    direction_score = compare_direction(
        current_context.get("goal_direction"),
        stored_context.get("goal_direction")
    )
    turn_score = compare_turn_patterns(
        current_context.get("turn_pattern", []),
        stored_context.get("turn_pattern", [])
    )
    length_score = compare_path_length(
        current_context.get("path_length", 0),
        stored_context.get("path_length", 0)
    )

    return (
        0.45 * patch_score +
        0.20 * direction_score +
        0.20 * turn_score +
        0.15 * length_score
    )

def best_matching_skill(current_context, skills):
    best_skill = None
    best_score = -1.0

    for skill in skills:
        stored_context = skill.get("start_context", {})
        score = similarity_score(current_context, stored_context)
        if score > best_score:
            best_score = score
            best_skill = skill

    return best_skill, best_score