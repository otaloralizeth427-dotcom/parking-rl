"""Punto de entrada: visualización, entrenamiento de experimentos, demo en terminal y evaluación."""

from __future__ import annotations

import argparse

from envs.levels import LEVEL_IDS, make_env
from envs.parking_env import ACTION_NAMES


def cmd_serve(args):
    import uvicorn

    from api.server import SimulationSession, create_app

    app = create_app(SimulationSession(args.level, args.sensor_noise))
    print(f"Visualización 3D en http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_train(args):
    from pipelines.train import main

    main([args.experiment])


def load_named_policy(name: str):
    """random, rule o el run_id de un modelo registrado."""
    if name == "random":
        return None
    if name == "rule":
        from agents.rule_based import RuleBasedPolicy
        return RuleBasedPolicy()
    from agents.registry import load_manifest, load_policy
    return load_policy(load_manifest(name))


def cmd_demo(args):
    policy = load_named_policy(args.policy)
    env = make_env(args.level)
    env.action_space.seed(args.seed)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\nEpisodio {ep + 1}: cupo {info['target_id']}, distancia {info['distance']:.1f} m")
        total = 0.0
        while True:
            action = int(env.action_space.sample()) if policy is None else policy.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if args.verbose:
                print(f"  t={info['steps']:3d} {ACTION_NAMES[action]:<10} v={info['speed']:+.2f} "
                      f"d={info['distance']:5.2f} θerr={info['heading_error']:+.2f} r={reward:+.2f}")
            if terminated or truncated:
                break
        outcome = "ÉXITO" if info["success"] else f"COLISIÓN ({info['collision_kind']})" \
            if info["collision"] else "TIEMPO AGOTADO"
        print(f"  → {outcome} en {info['steps']} pasos, recompensa {total:.1f}")


def cmd_evaluate(args):
    from pipelines.evaluate import main

    main(["--level", args.level])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="inicia la visualización 3D")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--level", choices=LEVEL_IDS, default=LEVEL_IDS[0])
    p.add_argument("--sensor-noise", type=float, help="sobrescribe el ruido del nivel")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("train", help="entrena un experimento de configs/experiments/")
    p.add_argument("experiment")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("demo", help="ejecuta episodios en la terminal")
    p.add_argument("--policy", default="random", help="random, rule o el run_id de un modelo registrado")
    p.add_argument("--level", choices=LEVEL_IDS, default=LEVEL_IDS[0])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("evaluate", help="compara referencias y modelos registrados con la suite del nivel")
    p.add_argument("--level", choices=LEVEL_IDS, default=LEVEL_IDS[0])
    p.set_defaults(func=cmd_evaluate)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if getattr(args, "episodes", 1) <= 0:
        raise SystemExit("--episodes debe ser positivo")
    args.func(args)
