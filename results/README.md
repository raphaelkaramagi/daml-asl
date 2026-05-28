# Evaluation and training outputs

Machine-readable metrics from `scripts/evaluate_models.py` and `scripts/train_landmark_nn.py`.

| File | Description |
|---|---|
| [`evaluation_results.json`](evaluation_results.json) | 28-photo test set metrics (ResNet + Landmark NN) |
| [`landmark_training.json`](landmark_training.json) | Latest landmark NN training summary |

The web demo reads a synced copy at [`web/public/evaluation-results.json`](../web/public/evaluation-results.json).

Regenerate after landmark retrain:

```bash
python scripts/run_landmark_pipeline.py
```

Or step by step:

```bash
python scripts/train_landmark_nn.py
python scripts/convert_models.py --landmark-only
python scripts/evaluate_models.py
python scripts/update_results_doc.py
```
