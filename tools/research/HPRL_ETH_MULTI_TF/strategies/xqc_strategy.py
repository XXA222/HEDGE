from .common import StrategySpec

SPEC = StrategySpec(
    name="HPRL_XQC_ETH",
    algorithm="xqc",
    description="XQC categorical critic with conditioning constraints; stability-oriented profile.",
    learning_rate=2.0e-4,
    hidden_dim=320,
    hidden_depth=3,
    gamma=0.995,
    tau=0.004,
    batch_size=512,
    replay_capacity=100_000,
    warmup_transitions=4096,
    tier_entropy_target_fraction=0.52,
    reward_overrides={
        "drawdown": 0.55,
        "downside": 0.25,
        "cvar": 0.18,
        "risk_projection": 0.10,
        "turnover": 0.0025,
    },
)
