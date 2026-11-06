"""Evaluación con suites fijas: compara referencias (aleatoria, regla) y modelos registrados.

Uso:  python -m pipelines.evaluate --level N1
Guarda en results/<nivel>/ los resultados por episodio, el resumen y las gráficas.
"""

from __future__ import annotations

import argparse
import json
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


def plot_comparison(comparison: dict, path: Path, title: str):
    keys = list(comparison)
    labels = [comparison[k]["label"].replace(" · ", "\n") for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

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
        ax.tick_params(axis="x", labelsize=8)
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
    fig, axes = plt.subplots(1, len(runs), figsize=(7 * len(runs), 4.5), squeeze=False)
    for ax, (run_id, curves) in zip(axes[0], runs):
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
    args = parser.parse_args(argv)
    compare_level(args.level)
    print(f"\nResultados en {RESULTS_DIR / args.level}/")


if __name__ == "__main__":
    main()
