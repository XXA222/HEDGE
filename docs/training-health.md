# Training Health telemetry

HEDGE observes whether a model is still learning without changing its activation functions,
losses, rewards, clipping or optimizer settings. The subsystem is enabled by default and writes
sampled metrics to the existing training logger/TensorBoard surface.

Supervised PyTorch trainers record `global_grad_norm`, per-layer gradient norms,
`near_zero_ratio`, and the actual `parameter_update_ratio` measured from weights immediately
before and after an optimizer step. Configure sampling and collapse thresholds with
`freqai.training_health`.

Stable-Baselines3 RL trainers additionally record `policy_grad_norm`, `value_grad_norm`,
`policy_entropy`, `action_saturation`, and `advantage_std`. Risk-Level RL uses the same contract
under the `train/risk_level_health` namespace. MultiDiscrete saturation means the fraction of
long/short actions at either action-space boundary; categorical saturation means the dominant
action share.

HPRL records actor/critic gradient norms, actual actor/critic update ratios, normalized empirical
policy entropy, action saturation, and one-step TD-advantage standard deviation. Its thresholds
are configured through the `health_*` fields in `HPRLTrainingConfig`. Telemetry uses deterministic
policy observations and therefore does not consume random samples or change the training
trajectory.

The rolling detector requires a complete window and a configurable consecutive-collapse
patience. It distinguishes policy and value gradient/update collapse, low-advantage collapse,
and joint low-entropy/high-saturation policy collapse. A detected condition emits a warning and
sets `training_health_collapsed=1`; it does not silently change the network or stop training.
