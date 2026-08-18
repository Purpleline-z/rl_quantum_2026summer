"""Epoch-checkpointed BT training used by bounded validation jobs."""
from __future__ import annotations

import os
import random
import time
import copy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); os.replace(temporary, path)


def train_with_epoch_checkpoints(exp, pair_ids, checkpoint_path: Path, phase: str, deadline: float | None,
                                 heartbeat_seconds: int = 60, checkpoint_enabled: bool = True,
                                 validation_split: str | None = None, early_stopping_patience: int | None = None):
    """Train with optional resumable epoch checkpoints. Returns (model, metrics, paused)."""
    rows = exp.rows_for(pair_ids)
    if rows.empty: raise ValueError("Cannot train with no labeled pairs.")
    model = exp.make_model(); optimizer = torch.optim.AdamW(model.parameters(), lr=exp.cfg.lr, weight_decay=exp.cfg.weight_decay)
    start_epoch, losses, epoch_metrics = 0, [], []
    best_validation, best_state, stale_epochs = float("-inf"), None, 0
    if checkpoint_enabled and checkpoint_path.exists():
        # Keep checkpoint metadata on CPU.  In particular, ``torch_rng_state``
        # belongs to the CPU generator; mapping the whole checkpoint to CUDA
        # turns it into a CUDA ByteTensor and ``torch.set_rng_state`` rejects it.
        # Model/optimizer loading moves parameter state to the model devices.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("phase") == phase and state.get("pair_ids") == list(pair_ids) and not state.get("partial_epoch", False):
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start_epoch, losses = int(state["completed_epochs"]), list(state.get("losses", []))
            epoch_metrics = list(state.get("epoch_metrics", []))
            best_validation = float(state.get("best_validation_accuracy", float("-inf")))
            best_state, stale_epochs = state.get("best_model"), int(state.get("stale_epochs", 0))
            if "torch_rng_state" in state:
                torch.set_rng_state(torch.as_tensor(state["torch_rng_state"], dtype=torch.uint8, device="cpu"))
    from pairwise_active_learning_pipeline import PairRows, TYPE_TO_INDEX, transform  # avoid a circular module import at file load time
    loader = DataLoader(PairRows(rows), batch_size=exp.cfg.train_batch_size, shuffle=True, num_workers=0)
    last_heartbeat = time.monotonic()
    for epoch in range(start_epoch, exp.cfg.epochs):
        model.train()
        epoch_losses = []
        for a, b, typ, winner, weight in loader:
            a, b, typ, weight = a.to(exp.device), b.to(exp.device), typ.to(exp.device), weight.to(exp.device)
            ra, rb = model(a), model(b); ix = torch.arange(len(typ), device=exp.device); xa, xb = ra[ix, typ], rb[ix, typ]
            terms = []
            for label in ("1", "2", "tie", "not_apply"):
                mask = torch.tensor([x == label for x in winner], device=exp.device)
                if not mask.any(): continue
                term = -F.logsigmoid(xa[mask] - xb[mask]) if label == "1" else -F.logsigmoid(xb[mask] - xa[mask]) if label == "2" else (xa[mask] - xb[mask]).abs() if label == "tie" else F.relu(xa[mask]) + F.relu(xb[mask])
                terms.append((term * weight[mask]).mean() * mask.float().mean())
            loss = sum(terms) if terms else torch.tensor(0., device=exp.device)
            anchor_types = list(exp.references)
            if len(anchor_types) > 1:
                preferred = random.choice(anchor_types); other = random.choice([x for x in anchor_types if x != preferred])
                p_path, o_path = random.choice(exp.references[preferred]), random.choice(exp.references[other]); tf = transform()
                with Image.open(p_path) as pi, Image.open(o_path) as oi:
                    pi, oi = tf(pi.convert("L")).unsqueeze(0).to(exp.device), tf(oi.convert("L")).unsqueeze(0).to(exp.device)
                index = TYPE_TO_INDEX[preferred]
                loss = loss + .25 * -F.logsigmoid(model(pi)[0, index] - model(oi)[0, index])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            scalar_loss = float(loss.detach().cpu()); losses.append(scalar_loss); epoch_losses.append(scalar_loss)
            now = time.monotonic()
            if checkpoint_enabled and now - last_heartbeat >= heartbeat_seconds:
                atomic_torch_save({"phase": phase, "pair_ids": list(pair_ids), "completed_epochs": epoch,
                                   "model": model.state_dict(), "optimizer": optimizer.state_dict(), "losses": losses,
                                   "torch_rng_state": torch.get_rng_state(), "config": exp.cfg.__dict__, "partial_epoch": True}, checkpoint_path)
                print(f"heartbeat phase={phase} epoch={epoch + 1}/{exp.cfg.epochs} checkpoint={checkpoint_path}", flush=True)
                last_heartbeat = now
            if deadline is not None and now >= deadline:
                return None, {"completed_epochs": epoch, "partial_epoch": True}, True
        metric = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan")}
        if validation_split:
            # Model selection is based exclusively on this validation split; outer_test is never queried here.
            validation = exp.evaluate(model.eval(), validation_split)
            metric["validation_split"] = validation_split
            metric["validation_accuracy"] = float(validation["test_accuracy"])
            if metric["validation_accuracy"] > best_validation:
                best_validation = metric["validation_accuracy"]
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                stale_epochs = 0
            else:
                stale_epochs += 1
            model.train()
        epoch_metrics.append(metric)
        if checkpoint_enabled:
            atomic_torch_save({"phase": phase, "pair_ids": list(pair_ids), "completed_epochs": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "losses": losses, "epoch_metrics": epoch_metrics, "best_validation_accuracy": best_validation, "best_model": best_state, "stale_epochs": stale_epochs, "torch_rng_state": torch.get_rng_state(), "config": exp.cfg.__dict__}, checkpoint_path)
        print(f"epoch progress phase={phase} epoch={epoch + 1}/{exp.cfg.epochs} saved_checkpoint={checkpoint_path}", flush=True)
        now = time.monotonic()
        if now - last_heartbeat >= heartbeat_seconds:
            suffix = f" checkpoint={checkpoint_path}" if checkpoint_enabled else " checkpointing=disabled"
            print(f"heartbeat phase={phase} epoch={epoch + 1}/{exp.cfg.epochs}{suffix}", flush=True); last_heartbeat = now
        if deadline is not None and now >= deadline:
            return None, {"completed_epochs": epoch + 1}, True
        if validation_split and early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            print(f"early stop phase={phase} at epoch={epoch + 1}; best validation accuracy={best_validation:.4f}", flush=True)
            break
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"loss": float(np.mean(losses)) if losses else float("nan"), "pairwise_accuracy": exp.pairwise_accuracy(model, rows),
                   "epoch_metrics": epoch_metrics, "best_validation_accuracy": best_validation if validation_split else None}, False
