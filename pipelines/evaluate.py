"""Evaluación con suites fijas: compara referencias (aleatoria, regla) y modelos registrados.

Uso:  python -m pipelines.evaluate --level N1
      python -m pipelines.evaluate --level N1 --new-final             # crea el test final reservado
      python -m pipelines.evaluate --level N1 --final <run_id> [...]  # lo usa una sola vez
Guarda en results/<nivel>/ los resultados por episodio, el resumen y las gráficas.

Test final reservado: sus semillas viven fuera del repo (PARKING_SECRET_DIR, por defecto
~/.parking-rl/). El repo solo guarda la huella SHA-256 de esas semillas, para comprobar después
que el test no cambió sin revelarlo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from envs.levels import LEVEL_IDS, make_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUITES_DIR = ROOT / "configs" / "suites"
RESULTS_DIR = ROOT / "results"
SECRET_DIR = Path(os.environ.get("PARKING_SECRET_DIR", Path.home() / ".parking-rl"))
OUTCOMES = ("success", "obstacle", "pedestrian", "boundary", "timeout")
OUTCOME_LABELS = {"success": "Éxito", "obstacle": "Choque con obstáculo", "pedestrian": "Atropello",
                  "boundary": "Salida del área", "timeout": "Tiempo agotado"}


def load_suite(suite_id: str) -> dict:
    path = SUITES_DIR / f"{suite_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe la suite {suite_id} ({path})")
    suite = json.loads(path.read_text())
    if suite["episodes"] <= 0:
        raise ValueError("La suite debe tener al menos un episodio")
    return suite


def suite_for_level(level_id: str) -> dict:
    return load_suite(f"{level_id}-suite-v1")


# ------------------------------------------------------------------ test final reservado
def commitment(secret: dict) -> str:
    return hashlib.sha256(json.dumps(secret, sort_keys=True).encode()).hexdigest()


def create_final_suite(level_id: str, episodes: int = 100) -> dict:
    suite_id = f"{level_id}-final-v1"
    public_path = SUITES_DIR / f"{suite_id}.json"
    if public_path.exists():
        raise FileExistsError(f"{suite_id} ya existe: el test final no se regenera")
    secret = {"suite_id": suite_id, "seed_start": 10_000_000 + secrets.randbelow(10_000_000),
              "salt": secrets.token_hex(16)}
    SECRET_DIR.mkdir(parents=True, exist_ok=True)
    (SECRET_DIR / f"{suite_id}.json").write_text(json.dumps(secret, indent=2))
    public = {"suite_id": suite_id, "level_id": level_id, "episodes": episodes, "deterministic": True,
              "reserved": True, "seed_commitment": commitment(secret),
              "description": "Test final reservado: las semillas no están en el repo; aquí solo su huella SHA-256."}
    public_path.write_text(json.dumps(public, indent=2) + "\n")
    return public


def final_suite(level_id: str) -> dict:
    public = load_suite(f"{level_id}-final-v1")
    secret_path = SECRET_DIR / f"{public['suite_id']}.json"
    if not secret_path.exists():
        raise FileNotFoundError(f"Las semillas reservadas no están en {secret_path}")
    secret = json.loads(secret_path.read_text())
    if commitment(secret) != public["seed_commitment"]:
        raise ValueError("Las semillas reservadas no coinciden con la huella publicada")
    return {**public, "seed_start": secret["seed_start"]}


def run_final(level_id: str, run_ids: list[str]) -> dict:
    """Evalúa la regla y los modelos elegidos en el test reservado; publica solo resultados sin semillas."""
    from agents.registry import load_manifest, load_policy

    suite = final_suite(level_id)
    policies = {"rule": baselines()["rule"]}
    for run_id in run_ids:
        manifest = load_manifest(run_id)
        policies[run_id] = (f"{manifest['agent'].upper()} · {run_id}", load_policy(manifest))
    report = {"level_id": level_id, "suite_id": suite["suite_id"], "seed_commitment": suite["seed_commitment"],
              "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "policies": {}}
    for key, (label, policy) in policies.items():
        result = evaluate(policy, suite)
        report["policies"][key] = {"label": label, **result["summary"],
                                   "outcomes_by_case": [c["outcome"] for c in result["cases"]]}
    out = RESULTS_DIR / level_id / "final_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print_table(report["policies"])
    return report


# ------------------------------------------------------------------ episodios
def run_case(env, policy, seed: int) -> dict:
    """Un episodio con un modelo congelado; policy=None es la política aleatoria."""
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    total, latencies = 0.0, []
    while True:
        start = time.perf_counter()
        action = int(env.action_space.sample()) if policy is None else policy.act(obs, greedy=True)
        latencies.append((time.perf_counter() - start) * 1000)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    outcome = "success" if info["success"] else info["collision_kind"] or "timeout"
    return {
        "seed": seed, "outcome": outcome, "success": info["success"],
        "terminated": terminated, "truncated": truncated,
        "steps": info["steps"], "time_s": round(info["steps"] * env.cfg.dt, 2),
        "return": round(total, 3), "final_distance_m": round(info["distance"], 3),
        "final_heading_error_deg": round(float(np.degrees(abs(info["heading_error"]))), 2),
        "latency_ms": round(float(np.mean(latencies)), 4),
    }


def evaluate(policy, suite: dict) -> dict:
    env = make_env(suite["level_id"])
    cases = [run_case(env, policy, suite["seed_start"] + i) for i in range(suite["episodes"])]
    return {"suite_id": suite["suite_id"], "level_id": suite["level_id"],
            "summary": summarize(cases), "cases": cases}


def summarize(cases: list[dict]) -> dict:
    counts = Counter(c["outcome"] for c in cases)
    n = len(cases)
    mean = lambda key: round(float(np.mean([c[key] for c in cases])), 3)  # noqa: E731
    return {
        "episodes": n,
        "success_rate": round(counts["success"] / n, 4),
        "outcomes": {o: counts.get(o, 0) for o in OUTCOMES},
        "mean_return": mean("return"),
        "mean_time_s": mean("time_s"),
        "mean_final_distance_m": mean("final_distance_m"),
        "mean_final_heading_error_deg": mean("final_heading_error_deg"),
        "mean_latency_ms": mean("latency_ms"),
    }


# ------------------------------------------------------------------ comparación por nivel
def baselines() -> dict:
    from agents.rule_based import RuleBasedPolicy
    return {"random": ("Aleatoria (referencia)", None), "rule": ("Regla (referencia)", RuleBasedPolicy())}


def registered_models(level_id: str) -> dict:
    from agents.registry import list_manifests, load_policy
    models = {}
    for manifest in list_manifests(level_id):
        if manifest["status"] == "completed":
            label = f"{manifest['agent'].upper()} · {manifest['run_id']}"
            models[manifest["run_id"]] = (label, load_policy(manifest))
    return models


def compare_level(level_id: str) -> dict:
    suite = suite_for_level(level_id)
    out_dir = RESULTS_DIR / level_id
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = {}
    for key, (label, policy) in {**baselines(), **registered_models(level_id)}.items():
        result = evaluate(policy, suite)
        (out_dir / f"eval_{key}.json").write_text(json.dumps(result, indent=1))
        comparison[key] = {"label": label, **result["summary"]}
    (out_dir / "comparison.json").write_text(json.dumps(
        {"level_id": level_id, "suite_id": suite["suite_id"], "policies": comparison}, indent=2))
    plot_comparison(comparison, out_dir / "comparison.png", f"Nivel {level_id} · suite {suite['suite_id']}")
    plot_curves(level_id, out_dir / "curves.png")
    print_table(comparison)
    return comparison


PALETTE = ["#9ca3af", "#f59e0b", "#3b82f6", "#8b5cf6", "#10b981", "#ef4444"]
OUTCOME_COLORS = {"success": "#22c55e", "obstacle": "#f97316", "pedestrian": "#ef4444",
                  "boundary": "#a855f7", "timeout": "#9ca3af"}


def short_label(key: str, label: str) -> str:
    """«N1-exp03-ppo-s42-64bf7433» → «exp03 PPO\ns42»; las referencias conservan su nombre."""
    parts = key.split("-")
    if len(parts) < 4:
        return label.split(" (")[0]
    return f"{parts[1]} {parts[2].upper()}\n{parts[3]}"


def plot_comparison(comparison: dict, path: Path, title: str):
    keys = list(comparison)
    labels = [short_label(k, comparison[k]["label"]) for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(max(16, 1.7 * len(keys) * 3 / 2), 5.5))

    # Resultado de los episodios por causa (barras apiladas)
    bottom = np.zeros(len(keys))
    for outcome in OUTCOMES:
        values = np.array([comparison[k]["outcomes"][outcome] for k in keys])
        axes[0].bar(labels, values, bottom=bottom, color=OUTCOME_COLORS[outcome], label=OUTCOME_LABELS[outcome])
        bottom += values
    axes[0].set(title="Resultado por episodio", ylabel="Episodios")
    axes[0].legend(frameon=False, fontsize=8)

    for ax, key, name in [(axes[1], "mean_time_s", "Tiempo promedio (s)"),
                          (axes[2], "mean_final_distance_m", "Error final de posición (m)")]:
        bars = ax.bar(labels, [comparison[k][key] for k in keys], color=PALETTE[:len(keys)])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_title(name)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8, rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(level_id: str, path: Path) -> bool:
    """Éxito durante el entrenamiento (con exploración) frente a validación (política congelada)."""
    from agents.registry import ROOT as REPO, list_manifests

    runs = [(m["run_id"], json.loads((REPO / m["config_uri"]).parent.joinpath("curves.json").read_text()))
            for m in list_manifests(level_id) if m["status"] == "completed"]
    if not runs:
        return False
    cols = min(3, len(runs))
    rows = -(-len(runs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    for ax in axes.flat[len(runs):]:
        ax.set_visible(False)
    for ax, (run_id, curves) in zip(axes.flat, runs):
        unit = "Pasos de interacción" if "-ppo" in run_id else "Episodios"
        train = curves["train"]
        ax.plot([p["step"] for p in train], [p["success_rate"] * 100 for p in train],
                color="#f59e0b", label="Entrenamiento (con exploración, media móvil 100)")
        val = curves["validation"]
        ax.plot([p["step"] for p in val], [p["success_rate"] * 100 for p in val], "o-",
                color="#3b82f6", label="Validación (política congelada, 20 episodios)")
        ax.set(title=run_id, xlabel=unit, ylabel="Éxito (%)", ylim=(-3, 103))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def print_table(comparison: dict):
    print(f"{'Política':<36}{'Éxito':>7}{'Choques':>9}{'Atrop.':>8}{'Salidas':>9}{'Tiempo':>8}{'Err pos':>9}")
    for item in comparison.values():
        o = item["outcomes"]
        print(f"{item['label']:<36}{item['success_rate']:>7.0%}{o['obstacle']:>9}{o['pedestrian']:>8}"
              f"{o['boundary']:>9}{item['mean_time_s']:>7.1f}s{item['mean_final_distance_m']:>8.2f}m")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", required=True, choices=LEVEL_IDS)
    parser.add_argument("--new-final", action="store_true", help="crea el test final reservado del nivel")
    parser.add_argument("--final", nargs="+", metavar="RUN_ID", help="evalúa estos modelos en el test reservado")
    args = parser.parse_args(argv)
    if args.new_final:
        suite = create_final_suite(args.level)
        print(f"Creado {suite['suite_id']} · huella {suite['seed_commitment'][:12]}… · semillas en {SECRET_DIR}/")
        return
    if args.final:
        run_final(args.level, args.final)
        print(f"\nResultado en {RESULTS_DIR / args.level}/final_test.json")
        return
    compare_level(args.level)
    print(f"\nResultados en {RESULTS_DIR / args.level}/")


if __name__ == "__main__":
    main()
