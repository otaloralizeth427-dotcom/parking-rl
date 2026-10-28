"""Agente Deep Q-Network implementado con NumPy (sin frameworks de deep learning).

Incluye memoria de experiencias, entrenamiento por lotes, red objetivo y Double DQN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from agent_base import MODELS_DIR, Agent


class MLP:
    """Perceptrón multicapa con ReLU y optimizador Adam."""

    def __init__(self, sizes: list[int], lr: float = 1e-3, seed: int | None = None):
        if len(sizes) < 2:
            raise ValueError("La red necesita al menos capa de entrada y de salida")
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.lr = lr
        self.params = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            w = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)).astype(np.float32)
            self.params += [w, np.zeros(n_out, dtype=np.float32)]
        self._m = [np.zeros_like(p) for p in self.params]
        self._v = [np.zeros_like(p) for p in self.params]
        self._t = 0

    def forward(self, x: np.ndarray, keep: bool = False):
        activations = [x]
        n_layers = len(self.params) // 2
        for i in range(n_layers):
            x = x @ self.params[2 * i] + self.params[2 * i + 1]
            if i < n_layers - 1:
                x = np.maximum(x, 0)
            activations.append(x)
        return (x, activations) if keep else x

    def backward(self, activations, grad_out) -> list[np.ndarray]:
        grads = [None] * len(self.params)
        g = grad_out
        for i in reversed(range(len(self.params) // 2)):
            a_in = activations[i]
            grads[2 * i] = a_in.T @ g
            grads[2 * i + 1] = g.sum(0)
            if i > 0:
                g = (g @ self.params[2 * i].T) * (activations[i] > 0)
        return grads

    def adam_step(self, grads, max_norm: float = 10.0, b1=0.9, b2=0.999, eps=1e-8):
        norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads))
        scale = min(1.0, max_norm / (norm + 1e-8))
        self._t += 1
        for p, g, m, v in zip(self.params, grads, self._m, self._v):
            g = g * scale
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            m_hat = m / (1 - b1 ** self._t)
            v_hat = v / (1 - b2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def copy_from(self, other: "MLP"):
        self.params = [p.copy() for p in other.params]


class ReplayBuffer:
    """Memoria circular de transiciones (s, a, r, s', terminado)."""

    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity debe ser positiva")
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self._idx = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        i = self._idx
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self._idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, batch_size)
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx]

    def __len__(self):
        return self.size


class DQNAgent(Agent):
    name = "neural"
    default_path = MODELS_DIR / "dqn.npz"

    def __init__(self, obs_dim: int = 14, n_actions: int = 5, hidden: tuple[int, ...] = (128, 128),
                 lr: float = 5e-4, gamma: float = 0.98, batch_size: int = 64,
                 buffer_size: int = 100_000, warmup: int = 1_000, train_every: int = 1,
                 target_sync: int = 1_000, reward_scale: float = 0.1, seed: int | None = None,
                 **kwargs):
        super().__init__(n_actions, seed=seed, **kwargs)
        if batch_size <= 0 or warmup < batch_size:
            raise ValueError("Se requiere batch_size > 0 y warmup >= batch_size")
        self.obs_dim = obs_dim
        self.hidden = tuple(hidden)
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup = warmup
        self.train_every = train_every
        self.target_sync = target_sync
        self.reward_scale = reward_scale   # escala las recompensas grandes (±100) para estabilizar
        sizes = [obs_dim, *hidden, n_actions]
        self.online = MLP(sizes, lr=lr, seed=seed)
        self.target = MLP(sizes, lr=lr, seed=seed)
        self.target.copy_from(self.online)
        self.memory = ReplayBuffer(buffer_size, obs_dim, seed=seed)
        self.steps = 0
        self.last_loss = 0.0

    def q_values(self, obs):
        return self.online.forward(np.asarray(obs, dtype=np.float32)[None])[0]

    def learn(self, obs, action, reward, next_obs, terminated):
        self.memory.add(obs, action, reward * self.reward_scale, next_obs, terminated)
        self.steps += 1
        if len(self.memory) >= self.warmup and self.steps % self.train_every == 0:
            self.last_loss = self._train_batch()
        if self.steps % self.target_sync == 0:
            self.target.copy_from(self.online)

    def _train_batch(self) -> float:
        obs, actions, rewards, next_obs, dones = self.memory.sample(self.batch_size)
        # Double DQN: la red online elige la acción y la red objetivo la evalúa
        best_next = self.online.forward(next_obs).argmax(1)
        next_q = self.target.forward(next_obs)[np.arange(len(actions)), best_next]
        targets = rewards + self.gamma * (1 - dones) * next_q

        q, activations = self.online.forward(obs, keep=True)
        rows = np.arange(len(actions))
        error = q[rows, actions] - targets
        # Pérdida de Huber: robusta frente a errores grandes
        grad = np.zeros_like(q)
        grad[rows, actions] = np.clip(error, -1.0, 1.0) / len(actions)
        self.online.adam_step(self.online.backward(activations, grad))
        abs_err = np.abs(error)
        return float(np.mean(np.where(abs_err < 1, 0.5 * error ** 2, abs_err - 0.5)))

    def save(self, path=None):
        path = Path(path or self.default_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"p{i}": p for i, p in enumerate(self.online.params)}
        np.savez_compressed(path, **arrays, sizes=np.array(self.online.sizes), gamma=self.gamma,
                            epsilon=self.epsilon, episodes=self.episodes_trained, lr=self.online.lr)
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or cls.default_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el modelo en {path}. Entrena primero el agente.")
        data = np.load(path)
        sizes = [int(s) for s in data["sizes"]]
        agent = cls(obs_dim=sizes[0], n_actions=sizes[-1], hidden=tuple(sizes[1:-1]),
                    lr=float(data["lr"]), gamma=float(data["gamma"]))
        agent.online.params = [data[f"p{i}"] for i in range(2 * (len(sizes) - 1))]
        agent.target.copy_from(agent.online)
        agent.epsilon = float(data["epsilon"])
        agent.episodes_trained = int(data["episodes"])
        return agent
