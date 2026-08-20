from .common import StrategySpec

SPEC = StrategySpec(
    name="HPRL_FastDSAC_ETH",
    algorithm="fast_dsac",
    description="Distributional maximum-entropy SAC; balanced exploration and downside control.",
    learning_rate=3.0e-4,
    hidden_dim=256,
    hidden_depth=3,
    gamma=0.992,
    tau=0.006,
    batch_size=512,
    replay_capacity=90_000,
    warmup_transitions=4096,
    tier_entropy_target_fraction=0.58,
    reward_overrides={
        "drawdown": 0.48,
        "downside": 0.24,
        "cvar": 0.16,
        "turnover": 0.0025,
    },
)
