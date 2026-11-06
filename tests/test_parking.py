import numpy as np
import pytest
from fastapi.testclient import TestClient

from agents.agent_base import run_episode, train
from agents.neural_q_learning import MLP, DQNAgent
from agents.q_learning import QLearningAgent
from api.server import SimulationSession, create_app
from envs.parking_env import BRAKE, FORWARD, LEFT, REVERSE, ParkingEnv, raycast_obbs


@pytest.fixture
def env():
    e = ParkingEnv("easy")
    e.reset(seed=0)
    return e


def place(env, x, y, heading, speed=0.0, steer=0.0):
    env.pos = np.array([x, y], dtype=float)
    env.heading, env.speed, env.steer = heading, speed, steer
    env._prev_potential = env._potential()


# ---------------------------------------------------------------- ambiente
def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["steps"] == 0 and not info["collision"] and not info["success"]
    assert env.speed == 0.0 and env.steer == 0.0
    assert not env._collides()


def test_spaces():
    env = ParkingEnv()
    assert env.action_space.n == 5
    assert env.observation_space.shape == (14,)
    assert env.observation_space.dtype == np.float32


def test_invalid_arguments():
    with pytest.raises(ValueError):
        ParkingEnv("imposible")
    with pytest.raises(ValueError):
        ParkingEnv(sensor_noise=-1)
    with pytest.raises(RuntimeError):
        ParkingEnv().step(0)
    env = ParkingEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(7)


def test_target_slot_is_never_occupied():
    env = ParkingEnv("hard")
    for seed in range(30):
        env.reset(seed=seed)
        cars = [o for o in env.obstacles if o["kind"] == "car"]
        assert all(np.linalg.norm(np.array(c["center"]) - env.target["center"]) > 1.0 for c in cars)


def test_forward_and_reverse_move_along_heading(env):
    place(env, 0.0, 0.0, 0.0)
    for _ in range(3):
        env.step(FORWARD)
    assert env.pos[0] > 0.3 and abs(env.pos[1]) < 1e-6 and env.speed > 0
    place(env, 0.0, 0.0, 0.0)
    for _ in range(3):
        env.step(REVERSE)
    assert env.pos[0] < -0.3 and env.speed < 0


def test_steering_turns_car_and_brake_stops(env):
    place(env, 0.0, 0.0, 0.0, speed=2.0)
    for _ in range(3):
        env.step(LEFT)
    assert env.steer > 0 and env.heading > 0.05
    for _ in range(3):
        env.step(BRAKE)
    assert env.speed == 0.0


def test_collision_with_parked_car_terminates(env):
    car = next(o for o in env.obstacles if o["kind"] == "car")
    place(env, *car["center"], np.pi / 2)
    _, reward, terminated, truncated, info = env.step(BRAKE)
    assert info["collision"] and terminated and not truncated
    assert reward <= env.cfg.r_collision + 5


def test_collision_with_lot_boundary(env):
    place(env, env.cfg.lot_width / 2 - 2.2, 0.0, 0.0, speed=3.0)
    terminated = False
    for _ in range(5):
        _, _, terminated, _, info = env.step(FORWARD)
        if terminated:
            break
    assert terminated and info["collision"]


def test_success_when_parked_aligned_and_stopped(env):
    tx, ty = env.target["center"]
    place(env, tx, ty, env.target["angle"])
    _, reward, terminated, _, info = env.step(BRAKE)
    assert info["success"] and info["in_slot"] and terminated
    assert reward > env.cfg.r_success / 2


def test_reverse_parking_also_counts(env):
    tx, ty = env.target["center"]
    place(env, tx, ty, env.target["angle"] + np.pi)
    _, _, _, _, info = env.step(BRAKE)
    assert info["success"]


def test_misaligned_or_moving_is_not_success(env):
    tx, ty = env.target["center"]
    place(env, tx, ty, env.target["angle"] + 0.6)
    assert not env._is_parked()
    place(env, tx, ty, env.target["angle"], speed=2.0)
    assert not env._is_parked()


def test_reward_prefers_approaching_target():
    env = ParkingEnv("easy")
    env.reset(seed=4)
    tx, ty = env.target["center"]
    heading = np.arctan2(ty - env.pos[1], tx - env.pos[0])
    start = env.pos.copy()
    place(env, *start, heading, speed=1.5)
    _, toward, *_ = env.step(FORWARD)
    place(env, *start, heading, speed=-1.5)
    _, away, *_ = env.step(REVERSE)
    assert toward > 0 > away


def test_truncation_after_max_steps():
    env = ParkingEnv("easy")
    env.reset(seed=0)
    truncated = False
    for _ in range(env.cfg.max_steps):
        _, _, terminated, truncated, _ = env.step(BRAKE)
        assert not terminated
    assert truncated


def test_seed_reproducibility():
    a, b = ParkingEnv(sensor_noise=0.1), ParkingEnv(sensor_noise=0.1)
    obs_a, _ = a.reset(seed=42)
    obs_b, _ = b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    for action in [0, 0, 2, 0, 3, 4, 1]:
        obs_a, r_a, *_ = a.step(action)
        obs_b, r_b, *_ = b.step(action)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert r_a == r_b
    obs_c, _ = a.reset(seed=43)
    assert not np.array_equal(obs_a, obs_c)


def test_sensor_noise_changes_readings():
    clean, noisy = ParkingEnv(), ParkingEnv(sensor_noise=0.2)
    obs_clean, _ = clean.reset(seed=5)
    obs_noisy, _ = noisy.reset(seed=5)
    assert not np.allclose(obs_clean[6:], obs_noisy[6:])


def test_raycast_distance():
    centers = np.array([[5.0, 0.0]])
    halves = np.array([[1.0, 1.0]])
    dirs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    hits = raycast_obbs(np.zeros(2), dirs, centers, halves, np.zeros(1), 10.0)
    np.testing.assert_allclose(hits, [4.0, 10.0])


def test_render_state_and_layout_are_serializable(env):
    import json
    json.dumps(env.render_state())
    json.dumps(env.layout())


# ---------------------------------------------------------------- agentes
def test_q_agent_greedy_selects_best_action():
    agent = QLearningAgent(seed=0)
    obs = ParkingEnv().reset(seed=0)[0]
    agent.q_table[agent.discretize(obs)] = [0, 0, 5, 0, 0]
    assert agent.act(obs, greedy=True) == 2


def test_epsilon_one_explores_all_actions():
    agent = QLearningAgent(seed=0, epsilon_start=1.0)
    obs = ParkingEnv().reset(seed=0)[0]
    assert {agent.act(obs) for _ in range(200)} == set(range(5))


def test_discretizer_index_in_range():
    agent = QLearningAgent()
    env = ParkingEnv()
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        assert 0 <= agent.discretize(obs) < agent.discretize.n_states


def test_dqn_action_selection():
    agent = DQNAgent(seed=0, hidden=(16,))
    obs = ParkingEnv().reset(seed=0)[0]
    q = agent.q_values(obs)
    assert q.shape == (5,)
    assert agent.act(obs, greedy=True) == int(np.argmax(q))


def test_mlp_gradient_matches_numeric():
    net = MLP([3, 5, 2], seed=0)
    net.params = [p.astype(np.float64) for p in net.params]
    rng = np.random.default_rng(1)
    x, w = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))
    _, acts = net.forward(x, keep=True)
    analytic = net.backward(acts, w)[0]
    numeric = np.zeros_like(net.params[0])
    for idx in np.ndindex(numeric.shape):
        orig = net.params[0][idx]
        net.params[0][idx] = orig + 1e-6
        plus = (net.forward(x) * w).sum()
        net.params[0][idx] = orig - 1e-6
        minus = (net.forward(x) * w).sum()
        net.params[0][idx] = orig
        numeric[idx] = (plus - minus) / 2e-6
    np.testing.assert_allclose(analytic, numeric, atol=1e-6)


def test_q_learning_basic_training(tmp_path):
    agent = QLearningAgent(seed=0, epsilon_decay=0.9)
    history = train(agent, ParkingEnv("easy"), episodes=20, seed=0)
    assert len(history) == 20
    assert agent.epsilon < 1.0 and agent.episodes_trained == 20
    assert np.any(agent.q_table != 0)
    loaded = QLearningAgent.load(agent.save(tmp_path / "q.npz"))
    np.testing.assert_array_equal(loaded.q_table, agent.q_table)


def test_dqn_basic_training(tmp_path):
    agent = DQNAgent(seed=0, hidden=(32, 32), warmup=64, batch_size=32, target_sync=50)
    before = [p.copy() for p in agent.online.params]
    train(agent, ParkingEnv("easy"), episodes=5, seed=0)
    assert len(agent.memory) > 64
    assert any(not np.allclose(a, b) for a, b in zip(before, agent.online.params))
    loaded = DQNAgent.load(agent.save(tmp_path / "dqn.npz"))
    obs = ParkingEnv().reset(seed=3)[0]
    np.testing.assert_allclose(loaded.q_values(obs), agent.q_values(obs), rtol=1e-6)


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        QLearningAgent.load(tmp_path / "nada.npz")
    with pytest.raises(FileNotFoundError):
        DQNAgent.load(tmp_path / "nada.npz")


def test_random_policy_episode_runs():
    result = run_episode(ParkingEnv("easy"), None, seed=0)
    assert set(result) >= {"return", "success", "collision", "steps"}


# ---------------------------------------------------------------- servidor
@pytest.fixture
def client():
    return TestClient(create_app(SimulationSession()))


def test_api_reset_and_step(client):
    reset = client.post("/api/reset", json={"seed": 1, "level": "N1"}).json()
    assert reset["type"] == "reset" and reset["layout"]["obstacles"]
    step = client.post("/api/step", json={"action": 0}).json()
    assert step["action_name"] == "Avanzar" and step["steps"] == 1
    for key in ("car", "sensors", "reward", "status", "distance"):
        assert key in step


def test_api_rejects_invalid_input(client):
    assert client.post("/api/step", json={"action": 99}).status_code == 400
    assert client.post("/api/agent", json={"agent": "nadie"}).status_code == 400
    assert client.post("/api/train", json={"episodes": 0}).status_code == 400
    assert client.post("/api/reset", json={"seed": -1}).status_code == 400


def test_websocket_commands(client):
    import json
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"cmd": "reset", "payload": {"seed": 2}}))
        assert json.loads(ws.receive_text())["type"] == "reset"
        ws.send_text(json.dumps({"cmd": "agent", "payload": {"agent": "random"}}))
        assert json.loads(ws.receive_text())["agent"] == "random"
        ws.send_text(json.dumps({"cmd": "step"}))
        assert json.loads(ws.receive_text())["type"] == "state"
        ws.send_text(json.dumps({"cmd": "nope"}))
        assert json.loads(ws.receive_text())["type"] == "error"


# ---------------------------------------------------------------- niveles
def test_levels_match_environment_version():
    from envs.levels import LEVEL_IDS, load_level, make_env
    from envs.parking_env import ENV_VERSION
    for level_id in LEVEL_IDS:
        assert load_level(level_id)["environment_version"] == ENV_VERSION
        obs, _ = make_env(level_id).reset(seed=0)
        assert obs.shape == (14,)
    with pytest.raises(ValueError):
        load_level("N9")


# ---------------------------------------------------------------- peatones (N3)
def test_levels_without_pedestrians_are_unchanged():
    from envs.levels import make_env
    a, b = make_env("N1"), ParkingEnv("normal", cones=0)
    np.testing.assert_array_equal(a.reset(seed=3)[0], b.reset(seed=3)[0])
    assert a.render_state()["pedestrians"] == []
    assert not [o for o in a.obstacles if o["kind"] == "cone"]


def test_pedestrians_walk_and_turn_around():
    env = ParkingEnv("hard", pedestrians=2)
    env.reset(seed=0)
    assert len(env.pedestrians) == 2
    place(env, -14.0, 0.0, 0.0)  # auto lejos para que nadie se detenga
    start = [p["y"] for p in env.pedestrians]
    for _ in range(5):
        env._move_pedestrians(env.cfg.dt)
    assert all(abs(p["y"] - y0) > 0.5 for p, y0 in zip(env.pedestrians, start))
    for _ in range(200):
        env._move_pedestrians(env.cfg.dt)
    assert all(abs(p["y"]) <= p["limit"] + 1e-9 for p in env.pedestrians)


def test_pedestrian_waits_next_to_car():
    env = ParkingEnv("hard", pedestrians=1)
    env.reset(seed=0)
    ped = env.pedestrians[0]
    place(env, ped["x"], ped["y"] + ped["dir"] * 1.6, np.pi / 2)  # auto atravesado en su camino
    y_before = ped["y"]
    env._move_pedestrians(env.cfg.dt)
    assert not ped["moving"] and ped["y"] == y_before


def test_hitting_pedestrian_is_a_collision():
    env = ParkingEnv("hard", pedestrians=1)
    env.reset(seed=0)
    ped = env.pedestrians[0]
    ped["speed"] = 0.0
    place(env, ped["x"], ped["y"], 0.0)
    _, _, terminated, _, info = env.step(BRAKE)
    assert terminated and info["collision"] and info["collision_kind"] == "pedestrian"


def test_sensors_detect_pedestrians():
    env = ParkingEnv("easy", pedestrians=1)
    env.reset(seed=0)
    ped = env.pedestrians[0]
    ped["speed"] = 0.0
    side = -1 if ped["x"] > 0 else 1   # el auto se ubica del lado con espacio, mirando al peatón
    place(env, ped["x"] + side * 5.0, ped["y"], 0.0 if side < 0 else np.pi)
    front = env._read_sensors()[0]
    assert abs(front - (5.0 - env._car_half[0] - env.cfg.pedestrian_half)) < 1e-6


# ---------------------------------------------------------------- referencias, paquetes y pipeline
def test_rule_policy_parks_sometimes():
    from agents.rule_based import RuleBasedPolicy
    from envs.levels import make_env
    env, policy = make_env("N1"), RuleBasedPolicy()
    results = [run_episode(env, policy, seed=100_000 + i) for i in range(10)]
    assert any(r["success"] for r in results)


def test_load_policy_rejects_tampered_model(tmp_path):
    from agents.registry import load_policy, sha256_file
    path = DQNAgent(seed=0, hidden=(8,)).save(tmp_path / "m.npz")
    manifest = {"run_id": "x", "status": "completed", "agent": "dqn",
                "model_uri": str(path), "model_sha256": sha256_file(path)}
    assert load_policy(manifest).q_values(np.zeros(14, np.float32)).shape == (5,)
    manifest["model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        load_policy(manifest)


def test_suite_evaluation_records_every_case():
    from pipelines.evaluate import evaluate, load_suite
    suite = dict(load_suite("N1-suite-v1"), episodes=3)
    result = evaluate(None, suite)
    assert len(result["cases"]) == 3 and result["summary"]["episodes"] == 3
    assert {"seed", "outcome", "latency_ms", "final_distance_m"} <= set(result["cases"][0])


@pytest.mark.parametrize("agent", ["dqn", "ppo"])
def test_pipeline_packages_a_run(tmp_path, monkeypatch, agent):
    import json
    import pipelines.train as pipeline
    from agents.registry import load_policy, sha256_file
    (tmp_path / "requirements.lock.txt").write_text("numpy\n")
    for name, value in {"ROOT": tmp_path, "ARTIFACTS_DIR": tmp_path / "artifacts",
                        "MANIFESTS_DIR": tmp_path / "manifests",
                        "MLFLOW_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
                        "MLFLOW_ARTIFACTS": tmp_path / "mlartifacts"}.items():
        monkeypatch.setattr(pipeline, name, value)
    (tmp_path / "manifests").mkdir()
    if agent == "dqn":
        config = pipeline.load_experiment("configs/experiments/N1-exp01-dqn.json")
        config.update({"budget": {"episodes": 3}, "validation": {"episodes": 2, "every": 2},
                       "hyperparameters": {"hidden": [16], "warmup": 64, "batch_size": 32}})
    else:
        config = pipeline.load_experiment("configs/experiments/N1-exp02-ppo.json")
        config.update({"budget": {"timesteps": 256}, "validation": {"episodes": 2, "every": 128},
                       "hyperparameters": {"n_steps": 128, "batch_size": 32, "n_epochs": 1, "net_arch": [16]}})
    config["experiment_id"] = f"test-{agent}"
    monkeypatch.setattr(pipeline, "suite_for_level", lambda level: {
        "suite_id": "mini", "level_id": level, "seed_start": 100000, "episodes": 2})
    manifest = pipeline.run_experiment(config)
    saved = json.loads((tmp_path / "manifests" / f"{manifest['run_id']}.json").read_text())
    model = tmp_path / saved["model_uri"]
    assert saved["status"] == "completed" and saved["model_sha256"] == sha256_file(model)
    for key in ("run_id", "level_id", "environment_version", "git_commit", "config_sha256",
                "model_uri", "model_sha256", "normalization_uri", "runtime_ref",
                "evaluation_suite_id", "seed", "parent_run_id"):
        assert key in saved
    saved["model_uri"] = str(model)
    assert load_policy(saved) is not None
