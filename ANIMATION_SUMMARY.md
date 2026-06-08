# Animation Feature Summary

## What Changed

The `test_llm_full_pipeline.py` has been enhanced with **step-by-step animated visualization** that displays the robot's decision-making process in clear, logical stages.

## Key Features Implemented

### 1. **Staged Console Output**
The console now shows 5 clear stages for each pipeline run:

```
STAGE 1: CORRUPTION SETUP
├─ Shows what type of corruption was applied
├─ Displays expected LLM behavior
└─ Visualizes original vs corrupted maps

STAGE 2: TIMESTEP ADVANCEMENT
├─ Shows temporal progression (t=0 → t=1 → t=2...)
├─ Auto-advances 2 timesteps for blocked scenarios
└─ No auto-advance for uncertain scenarios

STAGE 3: LLM QUERY & DECISION (Loop N)
├─ Shows current timestep and available data
├─ Displays LLM's action and reasoning
└─ Can appear multiple times if robot waits 2x then re-queries

STAGE 4: ACTION EXECUTION
├─ Shows how robot implements the decision
├─ For waits: advances to next timestep
├─ For replans: recalculates path
└─ For keep_moving: confirms path exists

STAGE 5: FINAL RESULT
├─ Shows pipeline completion status
├─ Indicates success/failure
└─ Displays total history entries
```

### 2. **Rich Decision Information**
Each decision now includes:
- **Action**: What the LLM recommends (wait, replan, keep_moving, etc.)
- **Reason**: Why the LLM chose this action
- **Context**: Current timestep, available timesteps, wait count

### 3. **Improved History Tracking**
History entries now include:
- Grid state at that moment
- Path (if found)
- Rich descriptive title
- LLM advice and reasoning (when applicable)

### 4. **Animated Visualization** (with `--window` flag)
Matplotlib display shows stages sequentially:
- Each stage is a separate figure window
- Groups related history entries together
- Includes descriptive text overlay on each subplot
- User can press Enter to advance or 'q' to quit

### 5. **Re-Query Logic Visualization**
The animation clearly shows:
- Initial LLM query
- Robot waits 1 timestep (wait count: 1/2)
- Robot waits 2nd timestep (wait count: 2/2 → triggers re-query)
- New LLM query with updated data
- Final action based on new context

## Example: Multi-Loop Scenario

**Command:**
```bash
python3 src/final_proj/test_llm_full_pipeline.py --mode uncertain_sparse --seed 3 --no-window
```

**Output shows:**
```
STAGE 3: LLM QUERY & DECISION (Loop 1)
Current timestep: t=0
→ LLM Decision: wait_and_reinspect
  Reason: obstacle pattern shifts across timesteps

STAGE 3: LLM QUERY & DECISION (Loop 2)
Current timestep: t=1
→ LLM Decision: wait_and_reinspect
  Wait count: 2/2
  ✓ Waited 2 timesteps - will query LLM again after advancing

STAGE 3: LLM QUERY & DECISION (Loop 3)
Current timestep: t=2
→ LLM Decision: plan_through
  Reason: uncertain region is surrounded by free space

STAGE 4: ACTION EXECUTION - KEEP MOVING
✓ Following current path to goal
```

This shows the robot:
1. Detecting obstacle pattern shifting at t=0 → waits
2. Observing again at t=1 → waits (count reaches 2)
3. Re-querying at t=2 → with new data, decides to proceed
4. Taking final action

## Console Output Improvements

### Before:
```
Auto-advancing to t=1
Auto-advancing to t=2
Step 1: t=2, action=wait, reason=...
-> wait and re-inspect (count=1), advancing to next timestep
-> no further timesteps available
Finished pipeline after 6 history entries
```

### After:
```
============================================================
STAGE 2: TIMESTEP ADVANCEMENT
============================================================
Auto-advancing 2 timesteps to gather temporal context...
  → Advanced to t=1
  → Advanced to t=2
Ready to query LLM with temporal data at t=2

============================================================
STAGE 3: LLM QUERY & DECISION (Loop 1)
============================================================
Current timestep: t=2
Remaining timesteps available: 1

→ LLM Decision:
  Action: wait
  Reason: blocked cell appears isolated and temporary

→ Waiting and re-inspecting...
  Wait count: 1/2
  ✗ No further timesteps available - finishing
```

## Usage

### With interactive animation:
```bash
python3 src/final_proj/test_llm_full_pipeline.py --mode blocked_moving --seed 99
```
- Shows matplotlib figure windows one stage at a time
- Press Enter to advance to next stage
- Type 'q' to quit early

### Console-only (no window):
```bash
python3 src/final_proj/test_llm_full_pipeline.py --mode blocked_moving --seed 99 --no-window
```
- Shows all stages in console output
- No interactive visualization
- Useful for logging/automation

## Technical Details

### Code Changes:

1. **Enhanced run() method:**
   - Added formatted stage headers with descriptive titles
   - Improved console logging with visual separators
   - Rich information about corruption types and expected answers
   - Better tracking of decision loops

2. **New display_history_animated() method:**
   - Groups history entries into logical stages
   - Creates one figure per stage (user-paced animation)
   - Includes stage descriptions and helpful text overlays
   - Interactive prompt system (Enter/Q)

3. **Improved _group_history_into_stages() method:**
   - Groups history entries by stage type
   - Recognizes keywords in titles to auto-categorize
   - Ensures proper stage ordering
   - Handles edge cases gracefully

4. **Better history entries:**
   - More descriptive titles
   - Corruption context and expected answers
   - LLM reasoning displayed as text overlay
   - Clear action descriptions

## Display Format

Each figure in the animation shows:
- **Title:** Stage name with clear action
- **Grid:** Color-coded map (gray=uncertain, white=free, black=obstacles)
- **Path:** Yellow line showing planned route
- **Markers:** Green star=start, Blue X=goal
- **Text overlay:** LLM advice and reasoning (in wheat-colored box)

## Testing Results

All scenarios tested successfully:
- ✅ `blocked_moving`: Auto-advances to t=2, queries with temporal data
- ✅ `blocked_permanent`: Detects permanent obstacle, replans
- ✅ `uncertain_sparse`: Queries at t=0, can re-query after waits
- ✅ `uncertain_clustered`: Handles clustered noise correctly
- ✅ Multi-loop scenarios: Shows re-query after 2 waits

## Example Output Comparison

### Scenario: blocked_moving with seed=99
```
Original path length: 142
Auto-advance to t=1, then t=2
Query at t=2 (has temporal context from 2 past timesteps)
LLM says: "wait" (temporary blockage)
Result: No more timesteps, pipeline ends
```

### Scenario: uncertain_sparse with seed=3
```
No auto-advance (uncertain scenario)
Loop 1: Query at t=0 → "wait"
Loop 2: Query at t=1 → "wait" (triggers re-query)
Loop 3: Query at t=2 → "plan_through" (proceed to goal)
Result: Success!
```

---

**The animated visualization makes the pipeline's decision-making process transparent and easy to understand!**
