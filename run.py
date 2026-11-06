"""Punto de entrada: servidor web, entrenamiento, demo en terminal y evaluación."""

from __future__ import annotations

import argparse
import json

from agents.agent_base import run_episode, train
from agents.registry import AGENT_CLASSES, TRAIN_DEFAULTS, history_path, load_or_create
from envs.levels import LEVEL_IDS, load_level
from envs.parking_env import ACTION_NAMES, DIFFICULTIES, ParkingEnv

DEFAULT_EPISODES = {"q_learning": 15_000, "neural": 4_000}


def cmd_serve(args):
    import uvicorn

    from api.server import SimulationSession, create_app

    app = create_app(SimulationSession(args.level or "N2", args.sensor_noise if args.sensor_noise_set else None))
    print(f"Visualización 3D en http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_train(args):
    episodes = args.episodes or DEFAULT_EPISODES[args.agent]
    if args.resume:
        agent, _ = load_or_create(args.agent)
    else:
        agent = AGENT_CLASSES[args.agent](seed=args.seed, **TRAIN_DEFAULTS[args.agent])
    env = ParkingEnv(args.difficulty, args.sensor_noise, pedestrians=args.pedestrians)
    history = train(agent, env, episodes, seed=args.seed, log_every=max(1, episodes // 20))
    path = agent.save()
    previous = json.loads(history_path(args.agent).read_text()) \
        if args.resume and history_path(args.agent).exists() else []
    history_path(args.agent).write_text(json.dumps(previous + history))
    print(f"Modelo guardado en {path}")


def cmd_demo(args):
    agent = None if args.agent == "random" else AGENT_CLASSES[args.agent].load()
    env = ParkingEnv(args.difficulty, args.sensor_noise, pedestrians=args.pedestrians)
    env.action_space.seed(args.seed)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\nEpisodio {ep + 1}: cupo {info['target_id']}, distancia {info['distance']:.1f} m")
        total = 0.0
        while True:
            action = int(env.action_space.sample()) if agent is None else agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if args.verbose:
                print(f"  t={info['steps']:3d} {ACTION_NAMES[action]:<10} v={info['speed']:+.2f} "
                      f"d={info['distance']:5.2f} θerr={info['heading_error']:+.2f} r={reward:+.2f}")
            if terminated or truncated:
                break
        outcome = "ÉXITO" if info["success"] else "COLISIÓN" if info["collision"] else "TIEMPO AGOTADO"
        print(f"  → {outcome} en {info['steps']} pasos, recompensa {total:.1f}")


def cmd_evaluate(args):
    from pipelines.evaluate import main

    main(["--episodes", str(args.episodes)])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def env_args(p):
        p.add_argument("--level", choices=LEVEL_IDS, help="nivel versionado; reemplaza --difficulty y --sensor-noise")
        p.add_argument("--difficulty", default="normal", choices=DIFFICULTIES)
        p.add_argument("--sensor-noise", type=float)
        p.add_argument("--pedestrians", type=int, default=0)

    p = sub.add_parser("serve", help="inicia la visualización 3D")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    env_args(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("train", help="entrena un agente")
    p.add_argument("--agent", required=True, choices=list(AGENT_CLASSES))
    p.add_argument("--episodes", type=int, help="por defecto: q_learning 15000, neural 4000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continúa desde el modelo guardado")
    env_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("demo", help="ejecuta episodios en la terminal")
    p.add_argument("--agent", default="random", choices=[*AGENT_CLASSES, "random"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    env_args(p)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("evaluate", help="compara las tres políticas y genera gráficas")
    p.add_argument("--episodes", type=int, default=200)
    p.set_defaults(func=cmd_evaluate)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if hasattr(args, "level"):
        args.sensor_noise_set = args.sensor_noise is not None
        if args.level:
            level = load_level(args.level)
            args.difficulty, args.pedestrians = level["difficulty"], level.get("pedestrians", 0)
            if not args.sensor_noise_set:
                args.sensor_noise = level["sensor_noise"]
        args.sensor_noise = args.sensor_noise or 0.0
    if getattr(args, "episodes", None) is not None and args.episodes <= 0:
        raise SystemExit("--episodes debe ser positivo")
    args.func(args)
