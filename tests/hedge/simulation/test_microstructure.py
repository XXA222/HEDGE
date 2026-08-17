from decimal import Decimal

from freqtrade.hedge.simulation.microstructure import MicrostructureState, simulate_taker_fill


def test_microstructure_fill_caps_liquidity_and_prices_latency() -> None:
    state = MicrostructureState(Decimal(99), Decimal(101), Decimal(2), Decimal(0), Decimal(500))
    fill = simulate_taker_fill(state, buy=True, quantity=Decimal(3))
    assert fill.filled_quantity == 2
    assert fill.fill_price > Decimal(101)
