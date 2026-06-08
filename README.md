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

## Quick validation commands

```bash
cd /home/sreshth/ros2_ws
python3 -m py_compile src/final_proj/test_llm_full_pipeline.py
python3 -m py_compile src/final_proj/full_integrated_test.py
```
