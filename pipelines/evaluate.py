"""Compara la política aleatoria, Q-learning y la red neuronal; genera gráficas en results/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agents.agent_base import run_episode  # noqa: E402
from agents.registry import AGENT_CLASSES, history_path  # noqa: E402
from envs.parking_env import DIFFICULTIES, ParkingEnv  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
COLORS = {"random": "#8a8f98", "q_learning": "#f59e0b", "neural": "#3b82f6"}
LABELS = {"random": "Aleatoria", "q_learning": "Q-learning", "neural": "Red neuronal"}
EVAL_SEED = 100_000  # semillas distintas a las del entrenamiento


def evaluate_policy(agent, difficulty: str, episodes: int, sensor_noise: float = 0.0) -> dict:
    env = ParkingEnv(difficulty, sensor_noise)
    env.action_space.seed(EVAL_SEED)
    runs = [run_episode(env, agent, seed=EVAL_SEED + i) for i in range(episodes)]
    return {
        "success_rate": float(np.mean([r["success"] for r in runs])),
        "collision_rate": float(np.mean([r["collision"] for r in runs])),
        "collisions": int(sum(r["collision"] for r in runs)),
        "mean_return": float(np.mean([r["return"] for r in runs])),
        "mean_steps": float(np.mean([r["steps"] for r in runs])),
        "mean_time_s": float(np.mean([r["steps"] for r in runs]) * env.cfg.dt),
        "final_position_error": float(np.mean([r["distance"] for r in runs])),
        "final_heading_error_deg": float(np.degrees(np.mean([r["heading_error"] for r in runs]))),
    }


def load_agents() -> dict:
    agents = {"random": None}
    for name, cls in AGENT_CLASSES.items():
        try:
            agents[name] = cls.load()
        except FileNotFoundError:
            print(f"Aviso: {name} no está entrenado; se omite (python run.py train --agent {name}).")
    return agents


METRICS_TO_PLOT = [
    ("success_rate", "Éxito (%)", 100),
    ("mean_return", "Recompensa promedio", 1),
    ("mean_time_s", "Tiempo promedio (s)", 1),
    ("collisions", "Número de colisiones", 1),
    ("final_position_error", "Error final de posición (m)", 1),
    ("final_heading_error_deg", "Error final de orientación (°)", 1),
]


def plot_comparison(results: dict, path: Path):
    difficulties = list(next(iter(results.values())).keys())
    agents = list(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    width = 0.8 / len(agents)
    x = np.arange(len(difficulties))
    for ax, (key, title, scale) in zip(axes.flat, METRICS_TO_PLOT):
        for i, agent in enumerate(agents):
            values = [results[agent][d][key] * scale for d in difficulties]
            bars = ax.bar(x + (i - (len(agents) - 1) / 2) * width, values, width,
                          color=COLORS[agent], label=LABELS[agent])
            ax.bar_label(bars, fmt="%.0f" if scale == 100 or key == "collisions" else "%.1f",
                         fontsize=8, padding=2)
        ax.set_title(title)
        ax.set_xticks(x, difficulties)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(agents), frameon=False)
    fig.suptitle("Comparación de políticas por dificultad", fontsize=14, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_training_curves(path: Path, window: int = 100) -> bool:
    histories = {n: json.loads(history_path(n).read_text())
                 for n in AGENT_CLASSES if history_path(n).exists()}
    if not histories:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, history in histories.items():
        kernel = np.ones(window) / window
        for ax, key in zip(axes, ("success", "return")):
            series = np.array([h[key] for h in history], dtype=float)
            if len(series) >= window:
                ax.plot(np.convolve(series, kernel, mode="valid") * (100 if key == "success" else 1),
                        color=COLORS[name], label=LABELS[name])
    axes[0].set(title=f"Éxito durante el entrenamiento (media móvil {window})",
                xlabel="Episodio", ylabel="Éxito (%)")
    axes[1].set(title="Retorno por episodio", xlabel="Episodio", ylabel="Retorno")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def print_table(results: dict):
    header = f"{'Agente':<14}{'Dific.':<8}{'Éxito':>7}{'Choques':>9}{'Recomp.':>9}" \
             f"{'Tiempo':>8}{'Err pos':>9}{'Err ang':>9}"
    print(header)
    print("-" * len(header))
    for agent, per_diff in results.items():
        for diff, m in per_diff.items():
            print(f"{LABELS[agent]:<14}{diff:<8}{m['success_rate']:>7.0%}{m['collisions']:>9d}"
                  f"{m['mean_return']:>9.1f}{m['mean_time_s']:>7.1f}s"
                  f"{m['final_position_error']:>8.2f}m{m['final_heading_error_deg']:>8.1f}°")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200, help="episodios por agente y dificultad")
    parser.add_argument("--difficulties", nargs="+", default=list(DIFFICULTIES), choices=DIFFICULTIES)
    parser.add_argument("--sensor-noise", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.episodes <= 0:
        parser.error("--episodes debe ser positivo")

    RESULTS_DIR.mkdir(exist_ok=True)
    results = {}
    for name, agent in load_agents().items():
        results[name] = {d: evaluate_policy(agent, d, args.episodes, args.sensor_noise)
                         for d in args.difficulties}
    print_table(results)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_comparison(results, RESULTS_DIR / "comparison.png")
    saved = ["metrics.json", "comparison.png"]
    if plot_training_curves(RESULTS_DIR / "training_curves.png"):
        saved.append("training_curves.png")
    print(f"\nGuardado en {RESULTS_DIR}/: {', '.join(saved)}")
    return results


if __name__ == "__main__":
    main()
