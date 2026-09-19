import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    from neuprint import Client, fetch_adjacencies, fetch_neurons, NC
    HAS_NEUPRINT = True
except ImportError:
    HAS_NEUPRINT = False

SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 0. GPU-NATIVE ONLINE OBSERVATION NORMALIZER
# ==============================================================================
class TorchObsNormalizer(nn.Module):
    """Vectorized PyTorch Online Observation Normalizer running directly on GPU."""

    def __init__(self, shape: tuple[int, ...], eps: float = 1e-8, clip: float = 5.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(shape, device=device))
        self.register_buffer("var", torch.ones(shape, device=device))
        self.register_buffer("count", torch.tensor(eps, device=device))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        norm = (x - self.mean) / (torch.sqrt(self.var) + self.eps)
        return torch.clamp(norm, -self.clip, self.clip)


# ==============================================================================
# 1. SURROGATE GRADIENT FUNCTION
# ==============================================================================
class SurrogateSpike(torch.autograd.Function):
    """Discrete spike step in forward pass, Fast Sigmoid surrogate in backward pass."""

    @staticmethod
    def forward(ctx, v: torch.Tensor, v_thresh: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.v_thresh = v_thresh
        return (v >= v_thresh).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v,) = ctx.saved_tensors
        v_thresh = ctx.v_thresh
        alpha = 2.0
        grad_v = grad_output / (1.0 + alpha * (v - v_thresh).abs()).square()
        return grad_v, None


act_spike = SurrogateSpike.apply


# ==============================================================================
# 2. CONNECTOME DATA LOADER
# ==============================================================================
def load_neuprint_connectome(n_neurons: int = 100):
    token = os.environ.get("NEUPRINT_TOKEN", "")

    if HAS_NEUPRINT and token:
        try:
            client = Client("https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token)
            neurons, _ = fetch_neurons(NC(rois=["EB", "FB"]))
            selected_neurons = neurons.head(n_neurons)
            body_ids = selected_neurons["bodyId"].tolist()
            _, conn_df = fetch_adjacencies(NC(bodyId=body_ids), NC(bodyId=body_ids))

            N = len(body_ids)
            W_bio = np.zeros((N, N), dtype=np.float32)
            id_to_idx = {b_id: idx for idx, b_id in enumerate(body_ids)}

            polarity = np.ones(N, dtype=np.float32)
            for idx, (_, row) in enumerate(selected_neurons.iterrows()):
                nt = str(row.get("ntPredicted", "")).lower()
                if "gaba" in nt or "glutamate" in nt:
                    polarity[idx] = -1.0

            for _, row in conn_df.iterrows():
                if row["bodyId_pre"] in id_to_idx and row["bodyId_post"] in id_to_idx:
                    src = id_to_idx[row["bodyId_pre"]]
                    dst = id_to_idx[row["bodyId_post"]]
                    W_bio[dst, src] += row["weight"] * polarity[src]

            max_val = np.abs(W_bio).max()
            if max_val > 0:
                W_bio /= max_val

            print(f"[Neuprint] Connectome loaded successfully ({N} neurons).")
            return torch.tensor(W_bio, dtype=torch.float32, device=device), N, body_ids
        except Exception as e:
            print(f"[Neuprint Warning] Connection failed ({e}). Generating synthetic connectome.")

    # Biologically plausible Dale's Law matrix: 80% Excitatory, 20% Inhibitory
    N = n_neurons
    W_bio = torch.randn(N, N, device=device) * 0.15
    inhib_mask = torch.rand(N, device=device) < 0.2
    W_bio[:, inhib_mask] = -torch.abs(W_bio[:, inhib_mask])
    W_bio[:, ~inhib_mask] = torch.abs(W_bio[:, ~inhib_mask])
    W_bio.fill_diagonal_(0.0)
    body_ids = [1000 + i for i in range(N)]
    return W_bio, N, body_ids


# ==============================================================================
# 3. NORMALIZED LIF ACTOR-CRITIC MODEL
# ==============================================================================
class SpikingActorCritic(nn.Module):

    def __init__(self, num_neurons: int, W_bio: torch.Tensor, substeps: int = 5):
        super().__init__()
        self.num_neurons = num_neurons
        self.substeps = substeps

        # Normalized LIF Dynamics Parameters
        self.v_thresh = 1.0
        self.decay_m = 0.85  # Membrane potential decay factor
        self.decay_s = 0.85  # Synaptic trace decay factor

        # Input Projection Layer
        self.fc_in = nn.Linear(8, num_neurons)
        nn.init.orthogonal_(self.fc_in.weight, gain=1.4)
        nn.init.zeros_(self.fc_in.bias)

        # Connectome Recurrent Layer
        self.W_rec = nn.Parameter(W_bio.clone())
        self.register_buffer("W_bio_ref", W_bio.clone().detach())

        # Dual Readout Heads (Synaptic Trace + Spike Rate)
        feat_dim = num_neurons * 2
        self.actor_head = nn.Linear(feat_dim, 4)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        self.critic_head = nn.Linear(feat_dim, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward_step(self, obs: torch.Tensor, v: torch.Tensor, syn_trace: torch.Tensor):
        is_single = (obs.dim() == 1)
        if is_single:
            obs = obs.unsqueeze(0)
            v = v.unsqueeze(0)
            syn_trace = syn_trace.unsqueeze(0)

        I_ext = self.fc_in(obs)
        spike_sum = torch.zeros_like(v)

        for _ in range(self.substeps):
            synaptic_input = torch.matmul(syn_trace, self.W_rec.t())
            v = v * self.decay_m + I_ext + synaptic_input
            spikes = act_spike(v, self.v_thresh)
            spike_sum = spike_sum + spikes

            v = v * (1.0 - spikes)  # Reset upon spiking
            syn_trace = syn_trace * self.decay_s + spikes

        spike_rate = spike_sum / float(self.substeps)
        feat = torch.cat([syn_trace, spike_rate], dim=-1)

        action_logits = self.actor_head(feat)
        state_value = self.critic_head(feat)

        if is_single:
            return action_logits.squeeze(0), state_value.squeeze(0), v.squeeze(0), syn_trace.squeeze(0)
        return action_logits, state_value, v, syn_trace


# ==============================================================================
# 4. SOTA PPO TRAINING PIPELINE
# ==============================================================================
def make_env(env_name: str, seed: int, rank: int):
    def _thunk():
        env = gym.make(env_name)
        env.action_space.seed(seed + rank)
        return env
    return _thunk


def train_ppo():
    print("=== STARTING SNN-PPO LUNAR LANDER TRAINING ===")
    W_bio, N, body_ids = load_neuprint_connectome(100)
    model = SpikingActorCritic(N, W_bio).to(device)

    NUM_ENVS = 8
    NUM_STEPS = 256
    TOTAL_TIMESTEPS = 819200
    MAX_UPDATES = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

    LR_INIT = 3e-4
    LR_FINAL = 3e-5
    ENTROPY_START = 0.02
    ENTROPY_FLOOR = 0.005

    BATCH_SIZE = 64
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    PPO_EPOCHS = 10
    CLIP_EPS = 0.15
    TARGET_KL = 0.015
    MAX_GRAD_NORM = 0.5
    VF_COEF = 0.5

    optimizer = optim.Adam(model.parameters(), lr=LR_INIT, eps=1e-5)

    env_name = "LunarLander-v3" if "LunarLander-v3" in gym.envs.registry else "LunarLander-v2"
    envs = gym.vector.SyncVectorEnv([make_env(env_name, SEED, i) for i in range(NUM_ENVS)])
    obs_normalizer = TorchObsNormalizer(shape=(8,)).to(device)

    recent_ep_rewards = deque(maxlen=100)
    ep_rewards_tracker = np.zeros(NUM_ENVS, dtype=np.float32)
    best_reward = -float("inf")

    raw_obs, _ = envs.reset(seed=SEED)
    raw_obs_tensor = torch.tensor(raw_obs, dtype=torch.float32, device=device)
    obs_normalizer.update(raw_obs_tensor)
    obs_tensor = obs_normalizer.normalize(raw_obs_tensor)

    v = torch.zeros((NUM_ENVS, N), device=device)
    syn_trace = torch.zeros((NUM_ENVS, N), device=device)

    for update in range(1, MAX_UPDATES + 1):
        progress = (update - 1) / float(MAX_UPDATES)
        lr_now = max(LR_FINAL, LR_INIT * (1.0 - progress))
        entropy_coef_now = max(ENTROPY_FLOOR, ENTROPY_START * (1.0 - progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        b_states = torch.zeros((NUM_STEPS, NUM_ENVS, 8), device=device)
        b_v_in = torch.zeros((NUM_STEPS, NUM_ENVS, N), device=device)
        b_syn_in = torch.zeros((NUM_STEPS, NUM_ENVS, N), device=device)
        b_actions = torch.zeros((NUM_STEPS, NUM_ENVS), device=device, dtype=torch.long)
        b_log_probs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_masks = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)

        # Rollout Phase
        for step in range(NUM_STEPS):
            b_states[step] = obs_tensor
            b_v_in[step] = v.clone()
            b_syn_in[step] = syn_trace.clone()

            with torch.no_grad():
                logits, value, v_next, syn_next = model.forward_step(obs_tensor, v, syn_trace)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_raw_obs, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            dones = terminated | truncated

            b_actions[step] = action
            b_log_probs[step] = log_prob
            b_values[step] = value.squeeze(-1)
            b_rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
            b_masks[step] = torch.tensor(1.0 - dones.astype(np.float32), device=device)

            ep_rewards_tracker += reward
            for env_idx, done in enumerate(dones):
                if done:
                    recent_ep_rewards.append(ep_rewards_tracker[env_idx])
                    ep_rewards_tracker[env_idx] = 0.0
                    v_next[env_idx] = 0.0
                    syn_next[env_idx] = 0.0

            v = v_next
            syn_trace = syn_next

            next_obs_tensor = torch.tensor(next_raw_obs, dtype=torch.float32, device=device)
            obs_normalizer.update(next_obs_tensor)
            obs_tensor = obs_normalizer.normalize(next_obs_tensor)

        # GAE Advantage Estimation
        with torch.no_grad():
            _, next_value, _, _ = model.forward_step(obs_tensor, v, syn_trace)
            next_value = next_value.squeeze(-1)

        advantages = torch.zeros_like(b_rewards, device=device)
        gae = torch.zeros(NUM_ENVS, device=device)

        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = b_masks[t]
                nextvalues = next_value
            else:
                nextnonterminal = b_masks[t]
                nextvalues = b_values[t + 1]

            delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
            gae = delta + GAMMA * GAE_LAMBDA * nextnonterminal * gae
            advantages[t] = gae

        returns = advantages + b_values

        # Buffer Flattening
        flat_states = b_states.reshape(-1, 8)
        flat_v_in = b_v_in.reshape(-1, N)
        flat_syn_in = b_syn_in.reshape(-1, N)
        flat_actions = b_actions.reshape(-1)
        flat_log_probs = b_log_probs.reshape(-1).detach()
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_values = b_values.reshape(-1).detach()

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # SGD Optimization Loop
        dataset_size = NUM_STEPS * NUM_ENVS
        indices = np.arange(dataset_size)
        total_loss_accum = 0.0
        num_batches = 0
        kl_stopped = False

        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_idx = indices[start:end]

                mb_states = flat_states[mb_idx]
                mb_v_in = flat_v_in[mb_idx]
                mb_syn_in = flat_syn_in[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_log_probs = flat_log_probs[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_advantages = flat_advantages[mb_idx]
                mb_values = flat_values[mb_idx]

                logits, new_values, _, _ = model.forward_step(mb_states, mb_v_in, mb_syn_in)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_log_probs
                ratios = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratios - 1) - log_ratio).mean()

                if approx_kl.item() > TARGET_KL:
                    kl_stopped = True
                    break

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                new_val = new_values.squeeze(-1)
                v_clipped = mb_values + torch.clamp(new_val - mb_values, -CLIP_EPS, CLIP_EPS)
                v_loss_unclipped = (new_val - mb_returns).pow(2)
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                bio_preservation_loss = 0.0005 * torch.norm(model.W_rec - model.W_bio_ref)
                total_loss = actor_loss + VF_COEF * critic_loss - entropy_coef_now * entropy + bio_preservation_loss

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                total_loss_accum += total_loss.item()
                num_batches += 1

            if kl_stopped:
                break

        avg_reward_100 = np.mean(recent_ep_rewards) if len(recent_ep_rewards) > 0 else -999.0
        avg_loss = total_loss_accum / max(1, num_batches)
        kl_status = " [KL Stop]" if kl_stopped else ""

        print(
            f"Update {update:03d}/{MAX_UPDATES} | LR: {lr_now:.2e} | "
            f"Entropy: {entropy_coef_now:.4f} | Avg Reward (100 Ep): {avg_reward_100:6.1f} | Loss: {avg_loss:.4f}{kl_status}"
        )

        if avg_reward_100 > best_reward and len(recent_ep_rewards) >= 20:
            best_reward = avg_reward_100
            torch.save(
                {"model_state": model.state_dict(), "body_ids": body_ids},
                "snn_ppo_lunar_winner.pt",
            )
            if avg_reward_100 >= 200.0:
                print(f"\nTARGET REACHED! 100-Episode Average Reward: {avg_reward_100:.2f}. Model saved.")

    envs.close()
    print(f"\nTraining Complete. Best 100-Ep Average Reward: {best_reward:.1f}")


if __name__ == "__main__":
    train_ppo()
