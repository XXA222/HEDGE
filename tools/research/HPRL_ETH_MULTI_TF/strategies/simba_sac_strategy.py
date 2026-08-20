from .common import StrategySpec

SPEC = StrategySpec(
    name="HPRL_SimbaSAC_ETH",
    algorithm="simba_sac",
    description="SimBa-style SAC; wider network with conservative gross-margin shaping.",
    learning_rate=2.5e-4,
    hidden_dim=384,
    hidden_depth=3,
    gamma=0.994,
    tau=0.005,
    batch_size=512,
    replay_capacity=100_000,
    warmup_transitions=4096,
    tier_entropy_target_fraction=0.55,
    reward_overrides={
        "drawdown": 0.52,
        "downside": 0.22,
        "cvar": 0.15,
        "gross_margin_risk": 0.11,
        "turnover": 0.0022,
    },
)
