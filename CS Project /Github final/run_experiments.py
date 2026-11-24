"""One manual runner for smoke, pilot, and full four-protocol experiments.

Contributor: Tejaswi Sapkota (sapkota3@illinois.edu).
Paper: https://proceedings.mlr.press/v297/nonaka26a.html
Uses the existing team task/model unchanged. No real-test evaluation here.
"""

import argparse
from contextlib import contextmanager
import inspect
import math
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiment_data import load_run, loader, setup_info, shuffled_cycles
from workflow import (DEVIATIONS, PROTOCOLS, STAGE_FRACTIONS, digest,
                      read_json, require, settings,
                      source_hashes, write_json)


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    elif device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if device == "mps":
        result["mps"] = torch.mps.get_rng_state()
    elif device.startswith("cuda"):
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device == "mps":
        torch.mps.set_rng_state(state["mps"])
    elif device.startswith("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def save_checkpoint(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(to_cpu(state), temporary)
    temporary.replace(path)


def optimizer_to(optimizer, device):
    # Adam's non-capturable step counter belongs on CPU; moments follow params.
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and name != "step":
                state[name] = value.to(device)


def binary_scores(target, probability):
    from sklearn.metrics import (average_precision_score, f1_score,
                                 roc_auc_score)

    target, probability = np.asarray(target), np.asarray(probability)
    require(set(np.unique(target)) == {0, 1}, "Evaluation requires both classes")
    require(np.isfinite(probability).all(), "Non-finite probabilities")
    positive = probability >= 0.5
    tp = int(np.sum((target == 1) & positive))
    tn = int(np.sum((target == 0) & ~positive))
    fp = int(np.sum((target == 0) & positive))
    fn = int(np.sum((target == 1) & ~positive))
    def ratio(a, b):
        return a / b if b else 0.0
    return {"roc_auc": float(roc_auc_score(target, probability)),
            "pr_auc": float(average_precision_score(target, probability)),
            "f1": float(f1_score(target, positive, zero_division=0)),
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def check_batch(batch):
    x, y = batch["signal"], batch["label"]
    require(tuple(x.shape) == (len(y), 1, 80000) and tuple(y.shape) == (len(y), 1),
            "Expected (B,1,80000) signals and (B,1) labels")
    require(torch.isfinite(x).all().item() and ((y == 0) | (y == 1)).all().item(),
            "Invalid waveform/label")


@torch.no_grad()
def evaluate(model, dataset, batch_size):
    model.eval()
    targets, probabilities, ids = [], [], []
    total, count = 0.0, 0
    for batch in loader(dataset, batch_size):
        check_batch(batch)
        out = model(signal=batch["signal"], label=batch["label"])
        value = float(out["loss"].item())
        require(math.isfinite(value), "Non-finite validation loss")
        n = len(batch["label"])
        total += value * n
        count += n
        targets.extend(out["y_true"].cpu().numpy().reshape(-1).tolist())
        probabilities.extend(out["y_prob"].cpu().numpy().reshape(-1).tolist())
        ids.extend(map(str, batch["patient_id"]))
    require(count > 0, "Empty evaluation partition")
    scores = binary_scores(targets, probabilities)
    scores.update(loss=total / count, recordings=count)
    return scores, {"patient_id": ids, "label": targets, "probability": probabilities}


def validate_artifacts(directory):
    """Check saved state without training or running predictions."""
    config = read_json(directory / "config.json")
    history = read_json(directory / "history.json")
    # last.pt contains RNG objects. Only load checkpoints you produced/trust.
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    require(last["config"] == config and last["history"] == history,
            "Checkpoint/config/history disagreement")
    require(last["epoch"] == len(history), "Epoch/history mismatch")
    if history:
        for key in ("steps", "real_seen", "synthetic_seen"):
            require(last[key] == history[-1][key], f"Counter mismatch: {key}")
        require(all(history[i]["epoch"] == i + 1 for i in range(len(history))),
                "Nonconsecutive history")
    evaluated = [h for h in history if "validation" in h]
    if evaluated:
        best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        chosen = min(evaluated, key=lambda h: h["validation"]["loss"])
        require(best["config"] == config and best["epoch"] == chosen["epoch"] and
                best["validation"] == chosen["validation"], "Best checkpoint mismatch")
        require(last["best_loss"] == best["validation"]["loss"], "Best loss disagreement")
    require(all(torch.isfinite(t).all().item() for t in last["model"].values()),
            "Non-finite saved weights")
    for state in last["optimizer"]["state"].values():
        require(int(state["step"].item()) == last["steps"], "Adam counter mismatch")
        require(all(torch.isfinite(state[k]).all().item() for k in ("exp_avg", "exp_avg_sq")),
                "Non-finite Adam state")
    require(config["test_evaluated"] is False, "Training evaluated test data")
    return last


@contextmanager
def output_lock(directory):
    lock = directory / ".running.lock"
    with lock.open("x") as stream:
        stream.write(f"pid={os.getpid()}\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def stage_config(protocol, index, data, options, source, provenance):
    domain = "syn" if protocol == "syn" else "real"
    weight = 1.0 if protocol == "syn" else data["real_weight"]
    budget = options["minutes"] * 60 * STAGE_FRACTIONS[protocol]
    return {"version": 1, "protocol": protocol, "mode": options["mode"],
            "run_index": index, "seed": index, "settings": options,
            "model": {**options["model"], "pos_weight": weight},
            "epoch_cap": options["stage_epochs"][protocol],
            "stage_budget_seconds": budget,
            "training_recordings": len(data[f"{domain}_train"]),
            "validation_recordings": len(data[f"{domain}_validation"]),
            "synthetic_pool_training_recordings": len(data["syn_train"]),
            "selection": f"minimum {domain} validation loss",
            "initialization": "synthetic best checkpoint" if source else "random",
            "pretraining": source,
            "mixed_loss": "0.5*real weighted BCE + 0.5*synthetic BCE",
            "mixed_epoch": "one real pass plus equal synthetic count; partial last batch kept",
            "provenance": provenance, "deviations": DEVIATIONS,
            "test_evaluated": False}


def run_stage(directory, protocol, index, data, options, provenance,
              resume=False, source_path=None, stop_after_epoch=None):
    from pyhealth.models import ResNet1D

    source = None
    if source_path:
        require(validate_artifacts(source_path.parent)["complete"],
                "Synthetic pretraining must finish before fine-tuning")
        source = {"sha256": digest(source_path),
                  "relative_path": "../syn/best.pt"}
        sc = torch.load(source_path, map_location="cpu", weights_only=True)
        require(sc["config"]["protocol"] == "syn" and
                sc["config"]["run_index"] == index and
                sc["config"]["provenance"] == provenance and
                sc["config"]["settings"] == options, "Wrong synthetic source checkpoint")
    config = stage_config(protocol, index, data, options, source, provenance)
    last_path = directory / "last.pt"
    if last_path.exists():
        require(resume, "Output exists. Use --resume only for your own trusted checkpoints")
        last = validate_artifacts(directory)
        require(last["config"] == config, "Resume settings/source/data/device differ")
        if last["complete"]:
            print(f"Already complete: run {index} {protocol}", flush=True)
            return True
    else:
        require(not directory.exists() or not any(directory.iterdir()),
                "Partial stage has no checkpoint. Inspect it and use a new output root")
        last = None
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", config)
    seed_all(index, options["device"])
    model_data = data["syn_train"] if protocol == "syn" else data["real_train"]
    model = ResNet1D(model_data, **config["model"])
    if source_path:
        model.load_pretrained(sc["model"], retain_head=True)
        require(abs(model.pos_weight.item() - data["real_weight"]) < 1e-6,
                "Synthetic transfer changed real class weighting")
        for key, value in sc["model"].items():
            if key != "pos_weight":
                torch.testing.assert_close(model.state_dict()[key], value)
    model.to(options["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"])
    history, start, steps, real_seen, syn_seen, bad = [], 0, 0, 0, 0, 0
    best_loss, elapsed = float("inf"), 0.0
    if last:
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        optimizer_to(optimizer, options["device"])
        history = last["history"]
        start, steps = last["epoch"], last["steps"]
        real_seen, syn_seen = last["real_seen"], last["synthetic_seen"]
        bad, best_loss, elapsed = last["bad_evaluations"], last["best_loss"], last["elapsed_seconds"]
        restore_rng(last["rng"], options["device"])
        del last

    domain = "syn" if protocol == "syn" else "real"
    train, validation = data[f"{domain}_train"], data[f"{domain}_validation"]
    cap, batch_size, device = config["epoch_cap"], options["batch_size"], options["device"]
    budget = config["stage_budget_seconds"]
    for epoch in range(start, cap):
        sync(device)
        started = time.perf_counter()
        model.train()
        if protocol == "mixed":
            half = batch_size // 2
            real_loader = loader(train, half, seed=index + epoch, shuffle=True)
            syn_order = shuffled_cycles(len(data["syn_train"]), len(train),
                                        index + epoch * 10000 + 5000)
            syn_loader = loader(data["syn_train"], half, indices=syn_order)
            batches = zip(real_loader, syn_loader)
            num_batches = math.ceil(len(train) / half)
        else:
            batches = ((batch, None) for batch in loader(
                train, batch_size, seed=index + epoch, shuffle=True))
            num_batches = math.ceil(len(train) / batch_size)
        total, seen, epoch_real, epoch_syn = 0.0, 0, 0, 0
        for batch, synthetic in tqdm(batches, total=num_batches,
                                      desc=f"run{index} {protocol} epoch{epoch + 1}"):
            check_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            n = len(batch["label"])
            if protocol == "mixed":
                check_batch(synthetic)
                require(len(synthetic["label"]) == n, "Mixed domain ratio differs")
                x = torch.cat((batch["signal"], synthetic["signal"]))
                y = torch.cat((batch["label"], synthetic["label"]))
                out = model(signal=x, label=y)
                real_loss = F.binary_cross_entropy_with_logits(
                    out["logit"][:n], out["y_true"][:n], pos_weight=model.pos_weight)
                syn_loss = F.binary_cross_entropy_with_logits(
                    out["logit"][n:], out["y_true"][n:])
                loss = 0.5 * real_loss + 0.5 * syn_loss
                epoch_real += n
                epoch_syn += n
                n *= 2
            else:
                out = model(signal=batch["signal"], label=batch["label"])
                loss = out["loss"]
                if protocol == "syn":
                    epoch_syn += n
                else:
                    epoch_real += n
            require(torch.isfinite(loss).item(), "Non-finite training loss")
            loss.backward()
            # Explicit finite-gradient guard also works on CUDA/MPS.
            require(all(p.grad is None or torch.isfinite(p.grad).all().item()
                        for p in model.parameters()), "Non-finite gradient")
            optimizer.step()
            total += float(loss.item()) * n
            seen += n
            steps += 1
        require(seen > 0, "No training batches")
        real_seen += epoch_real
        syn_seen += epoch_syn
        record = {"epoch": epoch + 1, "train_loss": total / seen, "steps": steps,
                  "real_seen": real_seen, "synthetic_seen": syn_seen,
                  "epoch_real_seen": epoch_real, "epoch_synthetic_seen": epoch_syn}
        if (epoch + 1) % options["val_every"] == 0 or epoch + 1 == cap:
            scores, _ = evaluate(model, validation, batch_size)
            record["validation"] = scores
            if scores["loss"] < best_loss:
                best_loss, bad = scores["loss"], 0
                save_checkpoint(directory / "best.pt", {
                    "model": model.state_dict(), "config": config,
                    "epoch": epoch + 1, "validation": scores})
            else:
                bad += 1
        sync(device)
        record["epoch_seconds_before_last_checkpoint"] = time.perf_counter() - started
        history.append(record)
        approximate_elapsed = elapsed + record["epoch_seconds_before_last_checkpoint"]
        reason = None
        if bad >= options["patience"]:
            reason = "early_stopping"
        elif epoch + 1 >= cap:
            reason = "epoch_cap"
        elif budget and "validation" in record and (
            approximate_elapsed + record["epoch_seconds_before_last_checkpoint"] >= budget
        ):
            reason = "pilot_time_budget"
        complete = reason is not None
        state = {"config": config, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "steps": steps, "real_seen": real_seen, "synthetic_seen": syn_seen,
                 "best_loss": best_loss, "bad_evaluations": bad, "history": history,
                 "rng": rng_state(device), "elapsed_seconds": approximate_elapsed,
                 "complete": complete, "completion_reason": reason}
        # Elapsed budget includes training, validation and any best-checkpoint
        # write; excludes this last-checkpoint write and JSON logging. Whole
        # invocation walltime is recorded separately by main().
        elapsed = approximate_elapsed
        save_checkpoint(last_path, state)
        write_json(directory / "history.json", history)
        write_json(directory / "stage_status.json", {
            "status": "complete" if complete else "partial", "reason": reason,
            "epoch": epoch + 1, "steps": steps, "real_seen": real_seen,
            "synthetic_seen": syn_seen, "elapsed_seconds": elapsed,
            "best_checkpoint_sha256": digest(directory / "best.pt")
            if (directory / "best.pt").exists() else None,
            "test_evaluated": False})
        print(f"Saved run{index}/{protocol}: epoch={epoch + 1}, "
              f"real={real_seen}, synthetic={syn_seen}, reason={reason}", flush=True)
        if complete:
            require((directory / "best.pt").exists(), "No validation-selected checkpoint")
            validate_artifacts(directory)
            if source_path:
                require(digest(source_path) == source["sha256"], "Source checkpoint changed")
            return True
        if stop_after_epoch and epoch + 1 >= stop_after_epoch:
            print("Stopped at requested epoch boundary; use same command with --resume.")
            return False
    raise RuntimeError("Unexpected stage termination")


def environment(device):
    from pyhealth.models import ResNet1D
    from pyhealth.models.protoecgnet import ResNet1dBackbone
    from pyhealth.tasks import BMDHSBinaryClassification
    from pyhealth.processors import AudioProcessor, BinaryLabelProcessor

    if device == "mps":
        require(torch.backends.mps.is_available(), "MPS unavailable; select CPU/CUDA")
    if device.startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA unavailable; install matching PyTorch build")
    return {"torch": str(torch.__version__), "numpy": str(np.__version__),
            "python": platform.python_version(), "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
            "team_source_hashes": {obj.__name__: digest(inspect.getfile(obj))
                                   for obj in (ResNet1D, ResNet1dBackbone,
                                               BMDHSBinaryClassification,
                                               AudioProcessor, BinaryLabelProcessor)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda:0"), default="cpu")
    parser.add_argument("--run-indices", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    for protocol in PROTOCOLS:
        parser.add_argument(f"--{protocol}-epochs", type=int)
    parser.add_argument("--val-every", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--minutes", type=float,
                        help="Combined approximate training-stage budget for pilot only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Operational interruption check; no extra validation forced")
    args = parser.parse_args()
    options = settings(args.mode, args.batch_size, args.epochs, args.val_every,
                       args.patience, args.minutes, args.run_indices)
    require(args.mode == "pilot" or options["minutes"] == 0,
            "Time budgets are pilot-only; full uses epoch caps/early stopping")
    require(args.mode != "pilot" or len(options["run_indices"]) == 1,
            "Pilot budget is for one paired run")
    require(not options["minutes"] or options["val_every"] == 1,
            "Timed pilot requires validation every epoch")
    options["stage_epochs"] = {}
    for protocol in PROTOCOLS:
        value = getattr(args, f"{protocol}_epochs")
        value = options["epochs"] if value is None else value
        require(value > 0, "Stage caps must be positive")
        options["stage_epochs"][protocol] = value
    if args.stop_after_epoch is not None:
        require(args.stop_after_epoch > 0, "Stop boundary must be positive")
    options["device"] = args.device
    setup, _, _, _ = setup_info(args.workspace)
    require(setup["mode"] == args.mode, "Use a workspace prepared for this exact mode")
    provenance = {"package_source_hashes": source_hashes(),
                  "real_plan_sha256": setup["real_plan_sha256"],
                  "synthetic_plan_sha256": setup["synthetic_plan_sha256"],
                  "real_csv_sha256": setup["real_csv_sha256"],
                  "real_audio_checksums_sha256": setup["real_audio_checksums_sha256"],
                  "synthetic_manifest_sha256": setup["synthetic_manifest_sha256"],
                  "environment": environment(args.device)}
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        require(args.resume, "Output is not empty; choose a new output or explicitly resume")
        require(read_json(args.output / "suite_config.json") == {
            "settings": options, "provenance": provenance}, "Suite configuration differs")
    else:
        require(not args.resume, "Resume requires a previous output")
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "suite_config.json", {
            "settings": options, "provenance": provenance})
    cache = args.workspace.resolve() / "cache/synthetic_temporary"
    cache.mkdir(parents=True, exist_ok=True)
    require(shutil.disk_usage(cache).free >= 8 * 1024 ** 3,
            "Allow at least 8 GiB free for processed data and outputs")
    tempfile.tempdir = str(cache)
    started = time.perf_counter()
    with output_lock(args.output):
        try:
            for index in options["run_indices"]:
                print(f"Loading paired run {index}; no test predictions...", flush=True)
                data = load_run(args.workspace, index, options["real_patients_per_class"])
                for protocol in PROTOCOLS:
                    stage = args.output / f"run_{index:02d}" / protocol
                    source = (args.output / f"run_{index:02d}/syn/best.pt"
                              if protocol == "syn2real" else None)
                    done = run_stage(stage, protocol, index, data, options, provenance,
                                     args.resume, source, args.stop_after_epoch)
                    if not done:
                        write_json(args.output / "suite_status.json", {
                            "status": "partial", "mode": args.mode, "run_index": index,
                            "protocol": protocol, "test_evaluated": False})
                        return
                del data
            stages = []
            for index in options["run_indices"]:
                for protocol in PROTOCOLS:
                    directory = args.output / f"run_{index:02d}" / protocol
                    last = validate_artifacts(directory)
                    require(last["complete"], "Incomplete stage")
                    stages.append({"run_index": index, "protocol": protocol,
                                   **read_json(directory / "stage_status.json")})
            write_json(args.output / "suite_status.json", {
                "status": "complete", "mode": args.mode,
                "run_indices": options["run_indices"], "stages": stages,
                "invocation_wall_seconds": time.perf_counter() - started,
                "test_evaluated": False,
                "uninterrupted_vs_resumed_equivalence_tested": False})
            print("PASS: all requested stages completed and artifacts checked.")
            print("No real-test predictions. Results:", args.output)
        except BaseException as error:
            write_json(args.output / "suite_status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "mode": args.mode, "error": str(error), "test_evaluated": False})
            raise


if __name__ == "__main__":
    main()
