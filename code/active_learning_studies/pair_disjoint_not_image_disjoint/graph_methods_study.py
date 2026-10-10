#!/usr/bin/env python3
"""Runner: new_methods_study (re-tuned schedule hooks, extra metrics) plus the graph-regularised design rules of graph_vopt.py."""
import judgment_unit_strategies as ju
import new_methods_study as nms   # registers new_methods rules and the extended evaluate; its __main__ block does not run on import
import graph_vopt as gv

ju.ROW_RULES.update(gv.NEW); ju.STRATEGY_FAMILY = ju.STRATEGY_FAMILY + tuple(n for n in gv.NEW if n not in ju.STRATEGY_FAMILY)

if __name__ == "__main__":
    nms.study.main()
