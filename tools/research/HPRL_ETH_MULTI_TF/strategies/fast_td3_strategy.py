from .common import StrategySpec

SPEC = StrategySpec(
    name="HPRL_FastTD3_ETH",
    algorithm="fast_td3",
    description="Deterministic distributional TD3; lower LR and stronger turnover discipline.",
    learning_rate=2.0e-4,
    hidden_dim=256,
    hidden_depth=3,
    gamma=0.995,
    tau=0.005,
    batch_size=512,
    replay_capacity=80_000,
    warmup_transitions=4096,
    reward_overrides={
        "drawdown": 0.50,
        "downside": 0.22,
        "turnover": 0.0030,
        "cvar": 0.14,
    },
)
