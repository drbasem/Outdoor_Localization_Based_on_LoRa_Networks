#!/bin/bash
# Full queue after switching XGBoost to tree_method="exact". Logged to results/queue.log.
set -x
cd "$(dirname "$0")/.."
ALL=$(python3 -c "print(' '.join(f'FV{i}' for i in range(1,17)))")
python3 scripts/run_experiments.py --source json --fv $ALL --models XGBoost --force
python3 scripts/run_experiments.py --fv FV1 FV2 --models XGBoost --force
python3 scripts/run_experiments.py --source json --fv FV4 --models Hybrid SVR
python3 scripts/run_experiments.py --source json --fv FV15 --models CNN
python3 scripts/run_experiments.py --fv FV1 FV2 --models Hybrid --force
python3 scripts/run_experiments.py --source json --fv $ALL --models k-NN XGBoost LightGBM ANN
echo QUEUE_DONE
