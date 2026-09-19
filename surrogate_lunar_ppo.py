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
    from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
    HAS_NEUPRINT = True
except ImportError:
    HAS_NEUPRINT = False

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 0. NORMALIZZATORE ONLINE DEGLI STATI (VECTORIZED)
# ==========================================
class ObsNormalizer:
    """Normalizza le osservazioni dell'ambiente per evitare saturazione dei neuroni SNN."""

    def __init__(self, shape, eps=1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = eps

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > 1 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def filter(self, x):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -5.0, 5.0)


# ==========================================
# 1. FUNZIONE SURROGATE GRADIENT
# ==========================================
class SurrogateSpike(torch.autograd.Function):
    """Calcola lo spike discreto nel forward pass e Fast Sigmoid nel backward pass."""

    @staticmethod
    def forward(ctx, v, v_thresh):
        ctx.save_for_backward(v)
        ctx.v_thresh = v_thresh
        return (v >= v_thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        v_thresh = ctx.v_thresh
        alpha = 2.0
        grad_v = grad_output / (1.0 + alpha * (v - v_thresh).abs()).square()
        return grad_v, None


act_spike = SurrogateSpike.apply


# ==========================================
# 2. CARICAMENTO CONNETTOMA BIOLOGICO
# ==========================================
def load_neuprint_connectome(n_neurons=100):
    token = "9f913383dbd800f5078cdf27745a225f5accfff5661b4aa8b975ac7ab3322e83"

    if HAS_NEUPRINT:
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

            print(f"[Neuprint] Connettoma caricato con successo ({N} neuroni).")
            return torch.tensor(W_bio, dtype=torch.float32, device=device), N, body_ids
        except Exception as e:
            print(f"[Neuprint Warning] Connessione non riuscita ({e}). Generazione matrice sintetizzata.")

    N = n_neurons
    W_bio = torch.randn(N, N, device=device) * 0.1
    body_ids = [1000 + i for i in range(N)]
    return W_bio, N, body_ids


# ==========================================
# 3. MODELLO LIF ACTOR-CRITIC DUAL-READOUT
# ==========================================
class SpikingActorCritic(nn.Module):

    def __init__(self, N, W_bio, substeps=5):
        super().__init__()
        self.N = N
        self.substeps = substeps

        # Parametri biologici del neurone LIF
        self.v_rest = -70.0
        self.v_thresh = -50.0
        self.v_reset = -75.0
        self.tau_m = 10.0
        self.tau_s = 10.0

        # Strato di proiezione con bilanciamento ortogonale
        self.fc_in = nn.Linear(8, N)
        nn.init.orthogonal_(self.fc_in.weight, gain=1.4)
        nn.init.zeros_(self.fc_in.bias)

        # Matrice del connettoma
        self.W_rec = nn.Parameter(W_bio.clone())
        self.W_bio_ref = W_bio.clone().detach()

        # Output Heads (Ingresso combinato: Synaptic Trace + Spike Rate)
        self.actor_head = nn.Linear(N * 2, 4)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        self.critic_head = nn.Linear(N * 2, 1)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward_step(self, obs, v, syn_trace):
        is_single = (obs.dim() == 1)
        if is_single:
            obs = obs.unsqueeze(0)
            v = v.unsqueeze(0)
            syn_trace = syn_trace.unsqueeze(0)

        I_ext = self.fc_in(obs) * 12.0
        spike_sum = torch.zeros_like(v)

        for _ in range(self.substeps):
            synaptic_input = torch.matmul(syn_trace, self.W_rec.t())
            dv = (-(v - self.v_rest) + I_ext + synaptic_input) / self.tau_m
            v = v + dv

            spikes = act_spike(v, self.v_thresh)
            spike_sum = spike_sum + spikes

            v = v * (1.0 - spikes) + self.v_reset * spikes
            syn_trace = syn_trace * (1.0 - 1.0 / self.tau_s) + spikes

        spike_rate = spike_sum / float(self.substeps)
        feat = torch.cat([syn_trace, spike_rate], dim=-1)

        action_logits = self.actor_head(feat)
        state_value = self.critic_head(feat)

        if is_single:
            return action_logits.squeeze(0), state_value.squeeze(0), v.squeeze(0), syn_trace.squeeze(0)
        return action_logits, state_value, v, syn_trace


# ==========================================
# 4. LOOP DI ADDESTRAMENTO PPO SOTA
# ==========================================
def make_env(env_name, seed, rank):
    def _thunk():
        env = gym.make(env_name)
        env.action_space.seed(seed + rank)
        return env
    return _thunk


def train_ppo():
    print("=== INIZIO TRAINING SNN-PPO SOTA (LUNAR LANDER +200 TARGET) ===")
    W_bio, N, body_ids = load_neuprint_connectome(100)
    model = SpikingActorCritic(N, W_bio).to(device)

    # Parametri SOTA di Schedulazione ed Esplorazione
    NUM_ENVS = 8
    NUM_STEPS = 256  # 8 envs * 256 steps = 2048 campioni per update
    TOTAL_TIMESTEPS = 819200  # 400 updates equivalenti
    MAX_UPDATES = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

    LR_INIT = 3e-4
    LR_FINAL = 5e-5
    ENTROPY_START = 0.020
    ENTROPY_FLOOR = 0.008

    BATCH_SIZE = 64
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    PPO_EPOCHS = 10
    CLIP_EPS = 0.15          # Clipping conservativo per gradienti SNN
    TARGET_KL = 0.015         # Early Stopping KL per prevenire regressione
    MAX_GRAD_NORM = 0.5       # Gradient norm clipping
    VF_COEF = 0.5

    optimizer = optim.Adam(model.parameters(), lr=LR_INIT, eps=1e-5)

    env_name = "LunarLander-v3" if "LunarLander-v3" in gym.envs.registry else "LunarLander-v2"
    envs = gym.vector.SyncVectorEnv([make_env(env_name, SEED, i) for i in range(NUM_ENVS)])
    obs_normalizer = ObsNormalizer(shape=(8,))

    recent_ep_rewards = deque(maxlen=100)
    ep_rewards_tracker = np.zeros(NUM_ENVS, dtype=np.float32)
    best_reward = -float("inf")

    # Inizializzazione stati ambienti
    raw_obs, _ = envs.reset(seed=SEED)
    obs_normalizer.update(raw_obs)
    norm_obs = obs_normalizer.filter(raw_obs)
    obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=device)

    v = torch.full((NUM_ENVS, N), model.v_rest, device=device)
    syn_trace = torch.zeros((NUM_ENVS, N), device=device)

    for update in range(1, MAX_UPDATES + 1):
        # Linear decay con limiti minimi (Floor)
        progress = (update - 1) / float(MAX_UPDATES)
        lr_now = max(LR_FINAL, LR_INIT * (1.0 - progress))
        entropy_coef_now = max(ENTROPY_FLOOR, ENTROPY_START * (1.0 - progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        # Buffer delle traiettorie
        b_states = torch.zeros((NUM_STEPS, NUM_ENVS, 8), device=device)
        b_v_in = torch.zeros((NUM_STEPS, NUM_ENVS, N), device=device)
        b_syn_in = torch.zeros((NUM_STEPS, NUM_ENVS, N), device=device)
        b_actions = torch.zeros((NUM_STEPS, NUM_ENVS), device=device, dtype=torch.long)
        b_log_probs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_masks = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)

        # --- FASE 1: ROLLOUT PARALLELIZZATO ---
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
                    # Reset selettivo dello stato SNN solo per gli ambienti terminati
                    v_next[env_idx] = model.v_rest
                    syn_next[env_idx] = 0.0

            v = v_next
            syn_trace = syn_next

            obs_normalizer.update(next_raw_obs)
            norm_next_obs = obs_normalizer.filter(next_raw_obs)
            obs_tensor = torch.tensor(norm_next_obs, dtype=torch.float32, device=device)

        # --- FASE 2: CALCOLO GAE ---
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

        # appiattimento buffer per il mini-batch SGD
        flat_states = b_states.reshape(-1, 8)
        flat_v_in = b_v_in.reshape(-1, N)
        flat_syn_in = b_syn_in.reshape(-1, N)
        flat_actions = b_actions.reshape(-1)
        flat_log_probs = b_log_probs.reshape(-1).detach()
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_values = b_values.reshape(-1).detach()

        # Normalizzazione vantaggi
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # --- FASE 3: OPTIMIZATION LOOP CON TARGET KL ---
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

                # Calcolo della KL Divergence per interrompere gli update distruttivi
                with torch.no_grad():
                    approx_kl = ((ratios - 1) - log_ratio).mean()

                if approx_kl.item() > TARGET_KL:
                    kl_stopped = True
                    break

                # Policy Loss
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value Loss con Clipping
                new_val = new_values.squeeze(-1)
                v_clipped = mb_values + torch.clamp(new_val - mb_values, -CLIP_EPS, CLIP_EPS)
                v_loss_unclipped = (new_val - mb_returns).pow(2)
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Preservazione della struttura connettomica biologica
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
            f"Entropia: {entropy_coef_now:.4f} | Reward Medio (100 Ep): {avg_reward_100:6.1f} | Loss: {avg_loss:.4f}{kl_status}"
        )

        # Salvataggio dinamico del miglior modello
        if avg_reward_100 > best_reward and len(recent_ep_rewards) >= 20:
            best_reward = avg_reward_100
            torch.save(
                {"model_state": model.state_dict(), "body_ids": body_ids},
                "snn_ppo_lunar_winner.pt",
            )
            if avg_reward_100 >= 200.0:
                print(f"\n TRAGUARDO RAGGIUNTO! Media 100 Episodi: {avg_reward_100:.2f}. Modello SOTA salvato.")

    envs.close()
    print(f"\nTraining Completato. Best 100-Ep Average Reward: {best_reward:.1f}")


if __name__ == "__main__":
    train_ppo()
