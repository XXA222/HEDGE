from __future__ import annotations

import unittest
from decimal import Decimal


class MemoryOptimizationContractAlignmentTest(unittest.TestCase):
    def test_native_hyperopt_uses_current_unstuck_field(self) -> None:
        from freqtrade.hedge.native.hyperopt import HedgeHyperoptSpace

        names = {space.name for space in HedgeHyperoptSpace().spaces}
        self.assertIn("unstuck_trigger_gross_exposure", names)
        self.assertNotIn("unstuck_threshold", names)

    def test_native_hyperopt_writes_fill_ratio_under_paper(self) -> None:
        from freqtrade.hedge.native.hyperopt import HedgeHyperoptSpace

        base = {"hedge": {"planner": {}, "paper": {}}}
        patched = HedgeHyperoptSpace.apply(
            base,
            {
                "max_fill_ratio_per_order": Decimal("0.25"),
                "unstuck_trigger_gross_exposure": Decimal("0.12"),
            },
        )
        self.assertEqual(
            patched["hedge"]["paper"]["max_fill_ratio_per_order"],
            "0.25",
        )
        self.assertEqual(
            patched["hedge"]["planner"]["unstuck_trigger_gross_exposure"],
            "0.12",
        )
        self.assertNotIn("max_fill_ratio_per_order", patched["hedge"])

    def test_research_optimizer_uses_serialized_paper_signal_names(self) -> None:
        from freqtrade.hedge.optimization.config_patch import ALLOWED_PARAMETER_PATHS

        self.assertIn("hedge.paper.long_signal", ALLOWED_PARAMETER_PATHS)
        self.assertIn("hedge.paper.short_signal", ALLOWED_PARAMETER_PATHS)
        self.assertNotIn("hedge.paper.default_long_signal", ALLOWED_PARAMETER_PATHS)
        self.assertNotIn("hedge.paper.default_short_signal", ALLOWED_PARAMETER_PATHS)

    def test_hedge_config_allows_research_optimization_section(self) -> None:
        from freqtrade.hedge.config import _HEDGE_ALLOWED_KEYS, _PLANNER_ALLOWED_KEYS

        self.assertIn("optimization", _HEDGE_ALLOWED_KEYS)
        self.assertIn("unstuck_limit_only", _PLANNER_ALLOWED_KEYS)


if __name__ == "__main__":
    unittest.main()
