# SFDE Package

## Description
The SFDE (Scientific Formula Dispatch Engine) runs a unified set of spatial, structural, cost, energy, MEP, and time-based formulas by **data affinity tags**. Each formula in `sfde_utility_foundation_extended.py` is decorated with an affinity label (e.g., "cost", "energy").

## How It Works
- Use `@affinity_tag("affinity_type")` to tag formulas.
- `runner_net.py` executes all matching formulas given a specific input and affinity type.

## Files
- `sfde_utility_foundation_extended.py` — all scientific formulas.
- `runner_net.py` — minimal affinity-based execution.
- `README_SFDE.txt` — this file.

## Example
Run cost and energy formulas with:

```bash
python runner_net.py
```
