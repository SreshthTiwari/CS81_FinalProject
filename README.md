# LLM-Guided Temporal Navigation

This repository contains simulation code for a robot navigation pipeline that is guided by an LLM. The system evaluates how the LLM decides whether the robot should keep moving, wait, or replan when the map observations are corrupted over time.

## Important files in this repo for testing

These are the most important test files to be able to replicate the video:

- `src/final_proj/test_llm_full_pipeline.py`: interactive end-to-end simulation with step-by-step decisions and optional visualization.
- `src/final_proj/full_integrated_test.py`: integrated test combining LLM planning with stored skill reuse and a visual history of stages.

## Prerequisites

1. Install Python 3.10 or newer.
2. Install required packages:

```bash
cd /home/sreshth/ros2_ws
python3 -m pip install numpy matplotlib
```

3. If you want to use a real LLM API instead of the local stub, set `GROQ_API_KEY` in your environment. Otherwise the local stub will run automatically.

## How to run `test_llm_full_pipeline.py`

This script runs a single full LLM-guided navigation scenario.
It is the primary interactive script for seeing the decision loop and how the planner responds to map corruption.

### What it does

- Loads the clean occupancy grid from `src/final_proj/final_proj/data/map.yaml`.
- Picks a random valid start and goal.
- Computes a clean A* path on the original map.
- Generates a temporal corruption sequence depending on the selected mode.
- Shows the following stages:
  1. corruption setup and expected LLM behavior
  2. timestep advancement / temporal observation
  3. LLM query and decision
  4. action execution (keep moving, wait, or replan)
  5. final result summary
- Optionally displays each stage with matplotlib if `--no-window` is not used.

### Step-by-step instructions

1. Open a terminal.
2. Change to the repository root:

```bash
cd /home/sreshth/ros2_ws
```

3. Run the script with a chosen mode and seed:

```bash
python3 src/final_proj/test_llm_full_pipeline.py --mode blocked_moving --seed 99
```

4. If the visualization window opens, click through each stage using the prompt shown in the terminal.
5. When the script finishes, it prints a JSON summary with whether the final path exists.

### Run without visualization

If you want to run the same pipeline without opening plots:

```bash
cd /home/sreshth/ros2_ws
python3 src/final_proj/test_llm_full_pipeline.py --mode blocked_moving --seed 99 --no-window
```

### Available modes

- `uncertain_sparse`: random sparse sensor noise appears on the path.
- `uncertain_clustered`: a cluster of uncertain/noisy cells blocks a region near the path.
- `blocked_moving`: a temporary moving obstacle crosses the path and may clear later.
- `blocked_permanent`: a permanent obstacle blocks the path and replanning is required.

## How to run `full_integrated_test.py`

This script runs an integrated test of LLM planning plus skill storage reuse.
It builds a visual history of stages that combines an initial stored route segment with a later replanning decision under corruption.

### What it does

- Loads the same clean map.
- Computes an initial A* path and splits it into stored checkpoints.
- Simulates a second start and a path to a checkpoint using stored skill reuse.
- Applies corruption to the final segment of the route based on the selected mode.
- Uses the LLM to decide whether to keep moving, wait/reinspect, or replan.
- Records a history of each step and visualizes the state, the path, checkpoints, and corrupted cells.
- Returns a final JSON summary showing success and total steps.

### Step-by-step instructions

1. Open a terminal.
2. Change to the repository root:

```bash
cd /home/sreshth/ros2_ws
```

3. Run the integrated test with a mode and optional seed:

```bash
python3 src/final_proj/full_integrated_test.py --mode uncertain_sparse --seed 1110
```

4. If visualization is available, press Enter after each figure to advance through the history.
5. At the end, the script prints a JSON summary containing:
   - `mode`
   - `success`
   - `final_path_exists`
   - `steps`
   - `last_title`

### Run without visualization

```bash
cd /home/sreshth/ros2_ws
python3 src/final_proj/full_integrated_test.py --mode blocked_permanent --seed 1234 --no-window
```

### Mode behavior in `full_integrated_test.py`

- `uncertain_sparse`: sparse uncertain cells appear on the remaining route.
- `uncertain_clustered`: a dense uncertain region appears near the remaining route.
- `blocked_moving`: a moving obstruction is injected on the remaining route.
- `blocked_permanent`: a permanent obstacle is placed on the remaining route.

## Detailed run examples

### Example 1: Animated full pipeline

```bash
cd /home/sreshth/ros2_ws
python3 src/final_proj/test_llm_full_pipeline.py --mode uncertain_clustered --seed 42
```

### Example 2: No-visualization integrated test

```bash
cd /home/sreshth/ros2_ws
python3 src/final_proj/full_integrated_test.py --mode blocked_permanent --seed 101 --no-window
```

### Example 3: Repeatable runs

To repeat experiments with the same random seed, use the same `--seed` value.

## Additional notes

- Both scripts use the local repository path, so run them from the repo root.
- If `matplotlib` is missing, use `--no-window` or install it with `python3 -m pip install matplotlib`.
- The two scripts share the same map and LLM planning components, but `full_integrated_test.py` adds skill storage reuse and a multi-stage history.

## What each script is for

### `src/final_proj/test_llm_full_pipeline.py`
Use this script when you want to:
- see the complete LLM-guided navigation pipeline
- observe how an LLM decision is made for a single scenario
- evaluate wait vs. replan behavior in a temporally corrupted map

### `src/final_proj/full_integrated_test.py`
Use this script when you want to:
- visualize an integrated sequence of stored skill reuse plus LLM replanning
- see how a previously computed route segment is reused as a skill
- observe multi-stage transitions from original route to corrupted final route

## How to run `test_skill_reuse_potential.py`

This script measures the reuse potential of stored navigation skills across multiple missions.
It answers the question: **How often can a new mission reuse path segments from a library of prior navigation experiences?**

### What it does

The test operates in two phases:

1. **Phase 1: Skill Library Construction**
   - Generates 10 initial missions (configurable) with random start and goal positions.
   - For each mission, computes an optimal A* path and segments it into chunks (~30 cells per segment).
   - Stores each segment (≥5 waypoints) in a skill library.

2. **Phase 2: Reuse Potential Analysis**
   - Generates 90 new missions (configurable) with random start and goal positions.
   - For each new mission's path, checks how many stored segments can be reused.
   - A segment is considered reusable if it matches ≥70% of the stored segment's waypoints.
   - Computes metrics on reuse rate, cells saved, and estimated computation savings.

### Key metrics

- **Reuse Rate** (%): Fraction of new missions that can reuse at least one stored segment.
- **Average Cells Reused per Mission**: Mean number of waypoints saved by reuse per mission.
- **Total Cells Reused**: Sum of reused waypoints across all new missions.
- **Estimated Computation Savings**: Rough estimate of avoided A* search operations (assumes cost ∝ segment_length^1.5).

### Step-by-step instructions

1. Open a terminal and navigate to the repository root:

```bash
cd /home/sreshth/ros2_ws
```

2. Run the test with default settings (100 total missions: 10 initial + 90 test):

```bash
python3 src/final_proj/test_skill_reuse_potential.py --seed 42
```

3. View the console output showing reuse statistics.

4. (Optional) Save detailed results to JSON:

```bash
python3 src/final_proj/test_skill_reuse_potential.py --num-missions 100 --initial-batch 10 --seed 42 --output reuse_results.json
```

### Arguments

- `--num-missions`: Total number of missions to analyze (default: 100).
- `--initial-batch`: Number of missions for the initial skill library (default: 10).
- `--seed`: Random seed for reproducibility (default: None).
- `--output`: Optional JSON file to save detailed per-mission results.

### Example output

```
Initial skill library size: 58 segments
Missions analyzed for reuse: 90
Missions with reusable segments: 59
Reuse rate: 65.6%
Average cells reused per mission: 132.1
Total cells reused across all missions: 11,879
Estimated computation savings: ~212,456 units
```

### Interpretation

A 65% reuse rate means roughly two-thirds of new missions can leverage skills from the library, avoiding redundant A* computation. For example, if a segment is 30 cells long and avoided computation cost is ~30^1.5 ≈ 164 operations, the savings compound across many missions.

## How to run `llm_experiments.py`

This script runs a batch of 40 LLM-guided navigation experiments, 10 per corruption mode.
It uses the exact same full-pipeline simulator as `test_llm_full_pipeline.py` to measure success rates and decision-making across multiple trials.

### What it does

- Runs trials across all four corruption modes: `uncertain_sparse`, `uncertain_clustered`, `blocked_moving`, `blocked_permanent`.
- For each mode, executes 10 independent missions (configurable with `--trials-per-mode`).
- Each trial runs the full LLM decision loop: corruption setup, LLM query, and action execution.
- Collects success/failure outcomes for each trial.
- Aggregates per-mode success rates and prints a JSON summary.
- Optionally exports detailed per-trial results to CSV or JSON.

### Key metrics

- **Success Rate per Mode**: Fraction of trials where the robot successfully reached the goal.
- **Total Successes**: Cumulative success count across all modes.
- **Final Path Exists**: Whether a valid path to the goal was available at the end of each trial.
- **History Length**: Number of decision-making stages per trial.

### Step-by-step instructions

1. Open a terminal and navigate to the repository root:

```bash
cd /home/sreshth/ros2_ws
```

2. Run 40 experiments (10 per mode) with a specified seed:

```bash
python3 src/final_proj/llm_experiments.py --no-window --trials-per-mode 10 --seed 1
```

3. The script prints a JSON summary with success rates per mode:

```json
{
  "total_trials": 40,
  "per_mode": {
    "uncertain_sparse": { "trials": 10, "success_count": 9, "success_rate": 0.9 },
    "uncertain_clustered": { "trials": 10, "success_count": 8, "success_rate": 0.8 },
    "blocked_moving": { "trials": 10, "success_count": 7, "success_rate": 0.7 },
    "blocked_permanent": { "trials": 10, "success_count": 6, "success_rate": 0.6 }
  },
  "total_successes": 30
}
```

4. (Optional) Export results to CSV:

```bash
python3 src/final_proj/llm_experiments.py --no-window --trials-per-mode 10 --seed 1 --csv-output experiment_results.csv
```

5. (Optional) Export results to JSON with full experiment details:

```bash
python3 src/final_proj/llm_experiments.py --no-window --trials-per-mode 10 --seed 1 --output experiment_results.json
```

### Arguments

- `--seed`: Base random seed for reproducibility (incremented per trial) (default: 0).
- `--noise-rate`: Base noise rate for corruption (default: 0.05).
- `--timesteps`: Number of temporal timesteps per scenario (default: 3).
- `--trials-per-mode`: Number of trials per corruption mode (default: 10).
- `--no-window`: Disable matplotlib visualization (recommended for batch runs).
- `--output`: Optional JSON file to save full experiment details.
- `--csv-output`: Optional CSV file to save per-trial results in tabular format.

### CSV output format

Each row represents a single trial with columns:
- `trial_id`: Unique trial identifier.
- `mode`: Corruption mode (`uncertain_sparse`, `uncertain_clustered`, `blocked_moving`, `blocked_permanent`).
- `seed`: Random seed used for this trial.
- `success`: Whether the trial succeeded (1 or 0).
- `final_path_exists`: Whether a valid path to goal existed at end of trial (1 or 0).
- `history_length`: Number of decision stages recorded during the trial.

### Use cases

- **Comparing LLM performance across corruption types**: Run multiple seeds and aggregate success rates per mode to see which scenarios are hardest for the LLM.
- **Benchmarking**: Use as a baseline to compare against alternative planning strategies (e.g., rule-based or different LLM models).
- **Reproducibility**: Use the same `--seed` value to reproduce exact trial sequences across multiple runs.

## Quick validation commands

```bash
cd /home/sreshth/ros2_ws
python3 -m py_compile src/final_proj/test_llm_full_pipeline.py
python3 -m py_compile src/final_proj/full_integrated_test.py
```
