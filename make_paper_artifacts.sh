#!/usr/bin/env bash
# Rebuild every figure and table of the paper from runs already on disk. No API calls.
#
#   ./make_paper_artifacts.sh            # uses ./data and ./output (or REL_DATA_DIR / REL_OUTPUT_DIR)
#   PYTHON=/path/to/python ./make_paper_artifacts.sh
#
# Figures land in output/figures/, LaTeX tables in output/tables/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1

step() { echo; echo "=== $*"; "$PY" "$@"; }

# Trajectory features for the reference-fix retrieval analysis
step behavior/leakage/extract_traj_features.py
step behavior/leakage/net_fetch_intent.py

# Behavior annotations -> behavior overview figure
step behavior/actonomy/make_behavior_overview.py

# Routing tables (self-routing, oracle, naive laddering)
step routing/make_self_routing.py
step routing/make_laddering.py

# Figures
step figures/make_instances_gained.py
step figures/make_solvable_cost.py --both
step figures/make_solvable_cost.py --corrected
step figures/make_solvable_cost.py --both --deepswe
step figures/make_effort_pair_cost.py
step figures/make_effort_allocation.py
step figures/create_swebenchpro_reference_panels.py
step figures/create_swebenchpro_example.py
step figures/deepswe/make_app_deepswe_instances_gained.py

# Tables
step figures/make_accuracy_tables.py
step figures/deepswe/make_deepswe_retry_ablation.py
