---
name: grow-manager
description: Orchestrates `grow-tech` and `grow-vision` analyses, aggregates subtask findings, and synthesizes a consolidated houseplant health report for each grow room.
version: 1.3.0
---

Purpose: Run subskills in parallel, gather their outputs, extract actionable insights, flag failures/incomplete data, and produce a single reconciled report per grow room.

Orchestration:
- Use `delegate_task` to run the following sub-tasks in parallel, batching into groups of 4 when appropriate.
  - /grow-tech analyze Grow Room 1
  - /grow-tech analyze Grow Room 2
  - /grow-vision analyze Grow Room 1
  - /grow-vision analyze Grow Room 2

Collection & Synthesis Rules:
- For each subtask, capture these fields: `key_findings` (3–6 bullet points), `metrics` (named values), and `confidence` (low/med/high)
- If a subtask fails or returns partial data, note the missing fields and continue synthesizing with available data. Include a `failed_tasks` list in the final report.
- Weight subtask findings by `confidence` when deriving recommendations.

Synthesis steps (per execution):
1. Group subtask outputs by Grow Room.
2. Extract top 5 `Key Findings` per room from combined subtasks.
4. Cross-room analysis: highlight asymmetries and shared infrastructure issues (Lung Room correlations).
5. Produce prioritized `Action Items` with estimated impact and urgency.

Required Final Report Schema (produce exactly):

### 🟢 Plant Analysis
- **Subtask Insights:**
  - **/grow-tech:** 
    — Key Findings: 
    — Key Metrics: Name: Value pairs for important sensor readings and trends
    — Confidence: [low|med|high]
  - **/grow-vision:** 
    — Key Findings:
    — Key Metrics: Name: Value pairs for important sensor readings and trends
    — Confidence: [low|med|high]
- **Current Lifecycle:** [Days since Start Date] | [Growth Stage/Phase]
- **Leaf VPD & Transpiration Health:** [Sensor Leaf VPD vs Target Range] — synthesized from subtask metrics
- **Trend Line Review:** [24-hour mean, min, max analysis]
- **Top 5 Key Findings:** [concise bullets synthesized across subtasks]
- **Action Items (prioritized):** [1. Quick wins, 2. Medium-term fixes, 3. Long-term projects]

### 🌐 Macro-Infrastructure Correlation
- **Equipment Usage:** [Correlation of equipment usage patterns with environmental conditions]
- **Cross-Room Diagnosis:** [Root-cause hypotheses supported by subtask data]

### ⚠️ Failed / Partial Tasks
- List failed or partial subtasks with missing fields and suggested re-run commands.
