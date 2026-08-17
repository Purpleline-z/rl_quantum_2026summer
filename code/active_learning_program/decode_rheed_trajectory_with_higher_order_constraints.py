#!/usr/bin/env python3
"""Soft, auditable Viterbi decoding for RHEED reconstruction trajectories.

This module is deliberately separate from metadata features and model training.  It
accepts precomputed per-frame state probabilities from a downstream classifier and
adds only explicitly configured trajectory-level physics preferences:

* a trajectory preferentially starts in (1 x 1);
* a transition from the initial (1 x 1) regime to another reconstruction
  preferentially passes through Bad; and
* after HTR, leaving HTR is penalised because HTR is expected to be final.

The rules are soft: no state is forbidden, and all changes to the framewise argmax
are written to the output for scientific review.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ("(1 x 1)", "Bad", "Twinned(2 x 1)", "c(6 x 2)", "(√13 x √13)", "HTR")
STATE_COLUMNS = {
    "(1 x 1)": "probability_1x1",
    "Bad": "probability_bad",
    "Twinned(2 x 1)": "probability_twinned",
    "c(6 x 2)": "probability_c6x2",
    "(√13 x √13)": "probability_sqrt13",
    "HTR": "probability_htr",
}
REQUIRED_COLUMNS = ("trajectory_id", "frame_index", "relative_image_path", *STATE_COLUMNS.values())


@dataclass(frozen=True)
class DecodingResult:
    states: list[str]
    score: float
    changed_frame_indices: list[int]
    penalties_applied: dict[str, int]


def load_configuration(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {"initial_state_not_1x1", "leave_htr", "skip_bad_before_other_reconstruction"}
    if set(config.get("penalty_rules", {})) != expected:
        raise ValueError("Configuration must provide exactly the three named penalty rules.")
    levels = config.get("sensitivity_levels", {})
    if not {"weak", "moderate", "strong"}.issubset(levels):
        raise ValueError("Configuration must provide weak, moderate, and strong sensitivity levels.")
    return config


def validate_score_table(scores: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in scores.columns]
    if missing:
        raise ValueError(f"Score table is missing required columns: {missing}")
    probabilities = scores.loc[:, list(STATE_COLUMNS.values())].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("All state probabilities must be finite and strictly positive.")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("Each frame's six state probabilities must sum to 1.0; scores are not silently recalibrated.")
    if scores[["trajectory_id", "frame_index"]].duplicated().any():
        raise ValueError("A trajectory contains duplicate frame_index values; resolve ordering ambiguity before decoding.")


def _penalty(config: dict, level: str, rule: str) -> float:
    """Return a non-negative log-score penalty for one named rule."""
    return float(config["sensitivity_levels"][level]["multiplier"]) * float(config["penalty_rules"][rule]["base_log_penalty"])


def decode_one_trajectory(probabilities: np.ndarray, config: dict, level: str) -> DecodingResult:
    """Decode one ordered trajectory using a two-state ``Bad has occurred`` memory.

    The dynamic-programming state is ``(current reconstruction, bad_has_occurred)``.
    It is the minimum additional history required to penalise a first transition from
    the 1x1 regime to another reconstruction when no Bad frame has appeared.  It is
    not a learned recurrent model and does not modify frame probabilities.
    """
    if probabilities.ndim != 2 or probabilities.shape[1] != len(STATES) or len(probabilities) == 0:
        raise ValueError("Expected a non-empty [frame, six-state] probability array.")
    emissions = np.log(probabilities)
    n_frames, n_states = emissions.shape
    # [state, seen_bad] where seen_bad is 0/1. Invalid state-memory combinations
    # remain -infinity, which makes backtracking unambiguous.
    scores = np.full((n_frames, n_states, 2), -np.inf, dtype=float)
    back_state = np.full((n_frames, n_states, 2), -1, dtype=int)
    back_bad = np.full((n_frames, n_states, 2), -1, dtype=int)
    first_penalty = _penalty(config, level, "initial_state_not_1x1")
    for current in range(n_states):
        current_seen_bad = int(STATES[current] == "Bad")
        scores[0, current, current_seen_bad] = emissions[0, current] - (0.0 if STATES[current] == "(1 x 1)" else first_penalty)

    for frame in range(1, n_frames):
        for previous in range(n_states):
            for seen_bad in range(2):
                previous_score = scores[frame - 1, previous, seen_bad]
                if not np.isfinite(previous_score):
                    continue
                for current in range(n_states):
                    current_seen_bad = int(seen_bad or STATES[current] == "Bad")
                    penalty = 0.0
                    if STATES[previous] == "HTR" and STATES[current] != "HTR":
                        penalty += _penalty(config, level, "leave_htr")
                    if not seen_bad and STATES[current] not in {"(1 x 1)", "Bad"}:
                        penalty += _penalty(config, level, "skip_bad_before_other_reconstruction")
                    candidate = previous_score + emissions[frame, current] - penalty
                    if candidate > scores[frame, current, current_seen_bad]:
                        scores[frame, current, current_seen_bad] = candidate
                        back_state[frame, current, current_seen_bad] = previous
                        back_bad[frame, current, current_seen_bad] = seen_bad

    final_state, final_seen_bad = np.unravel_index(np.argmax(scores[-1]), scores[-1].shape)
    decoded_indices = [final_state]
    current_state, current_seen_bad = final_state, final_seen_bad
    for frame in range(n_frames - 1, 0, -1):
        previous_state = back_state[frame, current_state, current_seen_bad]
        previous_seen_bad = back_bad[frame, current_state, current_seen_bad]
        decoded_indices.append(previous_state)
        current_state, current_seen_bad = previous_state, previous_seen_bad
    decoded_indices.reverse()
    raw_indices = probabilities.argmax(axis=1)
    penalties = {"initial_state_not_1x1": 0, "leave_htr": 0, "skip_bad_before_other_reconstruction": 0}
    if STATES[decoded_indices[0]] != "(1 x 1)":
        penalties["initial_state_not_1x1"] += 1
    seen_bad = STATES[decoded_indices[0]] == "Bad"
    for previous_index, current_index in zip(decoded_indices, decoded_indices[1:]):
        previous, current = STATES[previous_index], STATES[current_index]
        if previous == "HTR" and current != "HTR":
            penalties["leave_htr"] += 1
        if not seen_bad and current not in {"(1 x 1)", "Bad"}:
            penalties["skip_bad_before_other_reconstruction"] += 1
        seen_bad = seen_bad or current == "Bad"
    return DecodingResult(
        states=[STATES[index] for index in decoded_indices], score=float(scores[-1, final_state, final_seen_bad]),
        changed_frame_indices=np.flatnonzero(raw_indices != np.asarray(decoded_indices)).astype(int).tolist(),
        penalties_applied=penalties,
    )


def decode_score_table(scores: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_score_table(scores)
    ordered = scores.sort_values(["trajectory_id", "frame_index", "relative_image_path"]).reset_index(drop=True).copy()
    probability_matrix = ordered.loc[:, list(STATE_COLUMNS.values())].to_numpy(dtype=float)
    ordered["raw_framewise_argmax"] = [STATES[index] for index in probability_matrix.argmax(axis=1)]
    for level in ("weak", "moderate", "strong"):
        ordered[f"decoded_{level}_state"] = ""
        ordered[f"decoded_{level}_changed_from_raw"] = False
    summaries: list[dict] = []
    for trajectory_id, positions in ordered.groupby("trajectory_id", sort=False).groups.items():
        indices = list(positions)
        result_by_level: dict[str, DecodingResult] = {}
        for level in ("weak", "moderate", "strong"):
            result = decode_one_trajectory(probability_matrix[indices], config, level)
            result_by_level[level] = result
            ordered.loc[indices, f"decoded_{level}_state"] = result.states
            ordered.loc[indices, f"decoded_{level}_changed_from_raw"] = np.asarray(
                [i in result.changed_frame_indices for i in range(len(indices))], dtype=bool
            )
        summaries.append({
            "trajectory_id": trajectory_id,
            "frame_count": len(indices),
            **{f"{level}_decoded_path_score": result_by_level[level].score for level in result_by_level},
            **{f"{level}_changed_frame_count": len(result_by_level[level].changed_frame_indices) for level in result_by_level},
            **{f"{level}_{rule}_penalty_count": result_by_level[level].penalties_applied[rule] for level in result_by_level for rule in result_by_level[level].penalties_applied},
        })
    return ordered, pd.DataFrame(summaries)


def run_decoding(score_csv: Path, config_path: Path, output_root: Path) -> Path:
    config = load_configuration(config_path)
    score_table = pd.read_csv(score_csv)
    frames, summaries = decode_score_table(score_table, config)
    output_root.mkdir(parents=True, exist_ok=True)
    frames.to_csv(output_root / "per_frame_raw_and_soft_constrained_states.csv", index=False)
    summaries.to_csv(output_root / "per_trajectory_soft_constraint_summary.csv", index=False)
    provenance = {
        "input_score_csv": str(score_csv), "configuration": config,
        "decoder": "two-memory-state Viterbi; Bad occurrence memory only",
        "interpretation": "Soft constraints alter no model scores and forbid no path. Review raw and decoded columns together.",
    }
    (output_root / "soft_constraint_decoding_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-csv", required=True, type=Path, help="Classifier-produced six-state probabilities for ordered trajectory frames.")
    parser.add_argument("--config", required=True, type=Path, help="Higher-order constraint configuration JSON.")
    parser.add_argument("--output-root", required=True, type=Path, help="Small CSV/JSON output directory; no image or checkpoint is written.")
    args = parser.parse_args()
    print(run_decoding(args.score_csv, args.config, args.output_root))


if __name__ == "__main__":
    main()
