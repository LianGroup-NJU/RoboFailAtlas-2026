# Evaluation protocol

A MAE/RMSE use known scoring points. T reports ROC-AUC, balanced accuracy, and
success-class F1 on known terminal environment outcomes. Each model uses its
saved validation threshold; test labels never select a threshold.

The public evaluate command reports actions, failure types, outcomes, and
recovery at trajectory endpoints. Multilabel decisions use threshold 0.8.
Micro-P/R/F1 use all known category decisions. Macro-F1 averages only classes
with at least one known positive reference; known negative-only classes still
contribute false positives to micro metrics. Fields are masked independently.

Recovery uses three known classes. Prediction emits null when maximum softmax
probability is below 0.5. Null counts as a false negative for the known target
and assigns no false positive to another class. Macro-F1 uses positive-supported
classes. Accuracy and prediction coverage are also reported.

The low-level qwen_rm.structured_metrics module retains the historical 0.5
training-time diagnostic logger. It is not the endpoint paper-table evaluator.
Use python -m failatlas evaluate or scripts/evaluate_rm.sh for paper reporting.

Model metrics use fractions; PPO success rates use percentages. The optional
policy aggregation utility computes sample standard deviation across available
complete seeds, averaging tasks within each seed first. Missing seeds are not
imputed. Additional policy-seed runs were cancelled; the paper does not claim
multi-seed stability.

The release evaluator was replayed on saved predictions for 46,583 test scoring
points, including 46,530 valid A points and 2,951 endpoints. All 20 compared
task/diagnostic metrics agree with the original native report within 1e-12.
This CPU replay uses saved outputs; it is not a new GPU inference or training run.
