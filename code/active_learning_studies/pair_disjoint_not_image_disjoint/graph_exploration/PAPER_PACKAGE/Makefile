# One command regenerates every table, figure and delivered document from the stored result files (results/ is never modified).
#   make all      tables + figures + checked documents + tests
#   make tables   tables/*.csv|md and tables/numbers_*.json
#   make figures  figures/*.png|pdf (one script per figure)
#   make docs     check_prose.py (fails on any prose number that does not match a table) + build_docs.py
#   make test     unit tests (statistics helpers, cross-check against the studies' own documents, prose checker)
#   make clean    delete generated files (tables, figures, build, delivered documents)
PY ?= python3
.PHONY: all tables figures docs test clean
all: tables figures docs test

tables:
	$(PY) pp_seedblocks.py
	$(PY) pp_provenance.py
	$(PY) pp_confirmations.py
	$(PY) pp_curves.py
	$(PY) pp_ladder.py
	$(PY) pp_selection.py
	$(PY) pp_dose.py
	$(PY) pp_strata.py
	$(PY) pp_redteam_extra.py
	$(PY) pp_wcurse.py
	$(PY) pp_limits.py
	$(PY) pp_claims.py

figures: tables
	for f in figures/fig*.py; do $(PY) $$f || exit 1; done

docs: tables
	$(PY) check_prose.py
	$(PY) build_docs.py

test: tables
	$(PY) -m unittest discover -s tests

clean:
	find tables -type f ! -name 'prereg_order.*' ! -name 'numbers_prov.json' -delete   # the verified pre-registration order is kept (needs git history of all branches to recompute)
	rm -f figures/*.png figures/*.pdf build/* ABSTRACT_EN.md ADVISOR_SUMMARY_ZH.md
