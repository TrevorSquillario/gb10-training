---
name: grow-tech
description: Dual-environment execution loop to monitor, cross-reference, and optimize independent houseplant environments.
version: 1.2.0
---

# Skill: Dual-Context Houseplant Health Report

## Core Operational Directives

### 1. Unified Nomenclature Compliance
- You are optimizing two distinct spaces: **Plant 1** and **Plant 2**.
- Evaluate Plant 1 entirely, then evaluate Plant 2 entirely. Never average, blend, or bleed parameters between these distinct microclimates.
- For **Plant 1** details, load the reference: `skill_view("grow-tech", "references/PLANT1.md")`
- For **Plant 2** details, load the reference: `skill_view("grow-tech", "references/PLANT2.md")`
- For every report, load the reference: `skill_view("grow-tech", "references/LUNGROOM.md")`

### 2. Data Acquisition Protocol
- **Grow Log:** Retrieve the grow log notes for the last 7 days for each plant using `ha_get_history(entity_ids=["sensor.<sensor_name>"], start_time="7d")`
- **Real-time Sensors:** Get the current state for the sensors marked **Real-time Sensors** using home_assistant `ha_get_state(["sensor.<sensor_name>"], fields=["state"])`. Use a list to retrieve all real-time sensor states in a single API call for each plant.
- **Historical Sensors:** Get the last 24 hours history summarized from statistics for the sensors marked **Historical Sensors** using home_assistant
```
ha_get_history(source="statistics", entity_ids=["sensor.<sensor_name>"], start_time="24h", period="day", statistic_types=["mean", "min", "max"])
```

### 3. Advanced Botanical Insights
- **Direct Leaf VPD Utilization:** The provided VPD sensor metrics represent the actual *Leaf VPD* (accounting for canopy dynamics). Use these numbers directly to map against target thresholds for the specific `Growth Stage` (e.g., 0.8–1.2 kPa for active growth, 1.4–1.8 kPa for rest/dormancy).
- **Trend Diagnostics:** Analyze the 24-hour history for each historical sensor. Ensure the mean, min and max values are within healthy range.
- **Soil Moisture:** Analyze the soil moisture percentages. Cross-reference these percentages with the current growth stage to evaluate hydration needs. Determine if the current moisture percentage is optimal for root health, or if it indicates saturation or depletion thresholds that require triggering the irrigation systems.
- **Equipment Usage:** Analyze the exhaust fan usage patterns. Correlate excessive runtime with potential heat or humidity issues. Correlate insufficient runtime with potential air circulation or odor control issues.

### 4. Plant Details
- Soil Moisture Sensor Position: 4.5" Depth
- Plant 1: Tropical Plant Setup (High Humidity)
- Plant 2: Succulent/Cactus Setup (Low Humidity)

## Required Report Output Schema
For every execution cycle, format your output exactly as follows:

### 🟢 Plant x Analysis
- **Current Lifecycle:** [Days since Start Date] | [Growth Stage]
- **Leaf VPD & Transpiration Health:** [Sensor Leaf VPD vs Target Range]
- **Trend Line Review:** [Analysis of 24-hour mean, min, max values for historical sensors]
- **Action Items:** [Specific hardware or programmatic adjustments]

### 🌐 Macro-Infrastructure Correlation
- **Equipment Usage:** [Correlation of equipment usage patterns with environmental conditions]
- **Diagnosis:** [Cross-room data synthesis identifying root-cause infrastructure faults]