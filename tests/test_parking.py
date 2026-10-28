import numpy as np
import pytest
from fastapi.testclient import TestClient

from agent_base import run_episode, train
from neural_q_learning import MLP, DQNAgent
from parking_env import BRAKE, FORWARD, LEFT, REVERSE, ParkingEnv, raycast_obbs
from q_learning import QLearningAgent
from server import SimulationSession, create_app


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
    reset = client.post("/api/reset", json={"seed": 1, "difficulty": "easy"}).json()
    assert reset["type"] == "reset" and reset["layout"]["obstacles"]
    step = client.post("/api/step", json={"action": 0}).json()
    assert step["action_name"] == "Avanzar" and step["steps"] == 1
    for key in ("car", "sensors", "reward", "status", "distance"):
        assert key in step


def test_api_rejects_invalid_input(client):
    assert client.post("/api/step", json={"action": 99}).status_code == 400
    assert client.post("/api/agent", json={"agent": "nadie"}).status_code == 400
    assert client.post("/api/train", json={"agent": "random"}).status_code == 400
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
