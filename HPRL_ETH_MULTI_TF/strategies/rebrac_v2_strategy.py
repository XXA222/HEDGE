from .common import StrategySpec

SPEC = StrategySpec(
    name="HPRL_ReBRACv2_ETH",
    algorithm="rebrac_v2",
    description="Offline ReBRAC-v2 flow policy trained on a momentum/random behavior mixture.",
    learning_rate=2.0e-4,
    hidden_dim=320,
    hidden_depth=3,
    gamma=0.995,
    tau=0.005,
    batch_size=512,
    replay_capacity=80_000,
    warmup_transitions=4096,
    reward_overrides={
        "drawdown": 0.58,
        "downside": 0.26,
        "cvar": 0.18,
        "turnover": 0.0028,
        "hedge_overlap": 0.03,
    },
)
