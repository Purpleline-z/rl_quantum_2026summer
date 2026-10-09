#!/usr/bin/env python3
"""Runner for the new methods: judgment_unit_study with extra row rules registered (no change to the existing study code)."""
import judgment_unit_study as study
import judgment_unit_strategies as ju
import new_methods_strategies as nm

ju.ROW_RULES.update(nm.NEW); ju.STRATEGY_FAMILY = ju.STRATEGY_FAMILY + tuple(n for n in nm.NEW if n not in ju.STRATEGY_FAMILY)
_original_init = study.Context.__init__
def _init(self, *a, **k):
    _original_init(self, *a, **k); nm.CTX["ctx"] = self
study.Context.__init__ = _init

if __name__ == "__main__":
    study.main()
