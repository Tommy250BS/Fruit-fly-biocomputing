import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC
from torch.distributions import Categorical

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
# 1. FUNZIONE SURROGATE GRADIENT
# ==========================================
class SurrogateSpike(torch.autograd.Function):
    """Calcola lo spike discreto nel forward pass e un'approssimazione

    continua (Fast Sigmoid) nel backward pass per consentire BPTT.
    """

    @staticmethod
    def forward(ctx, v, v_thresh):
        ctx.save_for_backward(v, v_thresh)
        return (v >= v_thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, v_thresh = ctx.saved_tensors
        alpha = 2.0
        # Derivata della Fast Sigmoid: 1 / (1 + alpha * |v - v_thresh|)^2
        grad_v = grad_output / (1.0 + alpha * (v - v_thresh).abs()).square()
        return grad_v, None


act_spike = SurrogateSpike.apply


# ==========================================
# 2. CARICAMENTO CONNETTOMA BIOLOGICO
# ==========================================
def load_neuprint_connectome(n_neurons=100):
    token = "9f913383dbd800f5078cdf27745a225f5accfff5661b4aa8b975ac7ab3322e83"
    client = Client(
        "https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token
    )

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

    return torch.tensor(W_bio, dtype=torch.float32, device=device), N, body_ids


# ==========================================
# 3. MODELLO LIF ACTOR-CRITIC DIFFERENZIABILE
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

        # Strati di proiezione ed encodere
        self.fc_in = nn.Linear(8, N)

        # Matrice connettoma: inizializzata da W_bio con un vincolo di regolarizzazione Soft
        self.W_rec = nn.Parameter(W_bio.clone())
        self.W_bio_ref = W_bio.clone().detach()

        # Testate Actor e Critic basate sulle tracce sinaptiche
        self.actor_head = nn.Linear(N, 4)
        self.critic_head = nn.Linear(N, 1)

    def forward_step(self, obs, v, syn_trace):
        """Esegue substeps temporali di unroll BPTT per un singolo step di ambiente."""
        I_ext = self.fc_in(obs) * 10.0

        for _ in range(self.substeps):
            synaptic_input = torch.matmul(syn_trace, self.W_rec.t())
            dv = (-(v - self.v_rest) + I_ext + synaptic_input) / self.tau_m
            v = v + dv

            spikes = act_spike(v, self.v_thresh)

            # Reset del potenziale post-spike
            v = v * (1.0 - spikes) + self.v_reset * spikes

            # Integrale della traccia sinaptica
            syn_trace = syn_trace * (1.0 - 1.0 / self.tau_s) + spikes

        action_logits = self.actor_head(syn_trace)
        state_value = self.critic_head(syn_trace)

        return action_logits, state_value, v, syn_trace


# ==========================================
# 4. LOOP DI ADDESTRAMENTO PPO
# ==========================================
def train_ppo():
    print("=== INIZIO TRAINING SNN-PPO CON SURROGATE GRADIENT BPTT ===")
    W_bio, N, body_ids = load_neuprint_connectome(100)
    model = SpikingActorCritic(N, W_bio).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    env = gym.make("LunarLander-v3")

    MAX_EPISODES = 400
    STEPS_PER_UPDATE = 2048
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    PPO_EPOCHS = 10
    CLIP_EPS = 0.2

    best_reward = -float("inf")

    for episode in range(1, MAX_EPISODES + 1):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)

        # Inizializzazione stato neurale
        v = torch.full((N,), model.v_rest, device=device)
        syn_trace = torch.zeros(N, device=device)

        states, actions, log_probs, rewards, values, masks = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        ep_reward = 0.0

        for _ in range(STEPS_PER_UPDATE):
            logits, value, v, syn_trace = model.forward_step(
                obs, v.detach(), syn_trace.detach()
            )
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value.squeeze(-1))
            rewards.append(reward)
            masks.append(1.0 - float(done))

            ep_reward += reward
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

            if done:
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=device)
                v = torch.full((N,), model.v_rest, device=device)
                syn_trace = torch.zeros(N, device=device)

        # Calcolo GAE (Generalized Advantage Estimation)
        with torch.no_grad():
            _, next_value, _, _ = model.forward_step(obs, v, syn_trace)
            next_value = next_value.squeeze(-1)

        returns = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            delta = (
                rewards[i]
                + GAMMA * next_value * masks[i]
                - values[i].detach()
            )
            gae = delta + GAMMA * GAE_LAMBDA * masks[i] * gae
            next_value = values[i].detach()
            returns.insert(0, gae + values[i].detach())

        b_states = torch.stack(states)
        b_actions = torch.stack(actions)
        b_log_probs = torch.stack(log_probs).detach()
        b_returns = torch.tensor(returns, device=device)
        b_advantages = b_returns - torch.stack(values).detach()
        b_advantages = (b_advantages - b_advantages.mean()) / (
            b_advantages.std() + 1e-8
        )

        # Aggiornamento PPO
        for _ in range(PPO_EPOCHS):
            v_dummy = torch.full((N,), model.v_rest, device=device)
            syn_dummy = torch.zeros(N, device=device)

            logits, new_values, _, _ = model.forward_step(
                b_states, v_dummy, syn_dummy
            )
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_actions)

            ratios = torch.exp(new_log_probs - b_log_probs)
            surr1 = ratios * b_advantages
            surr2 = (
                torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
                * b_advantages
            )

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(-1), b_returns)

            # Penale L2 per mantenere la matrice vicina alla topologia biologica originale W_bio
            bio_preservation_loss = 0.01 * torch.norm(
                model.W_rec - model.W_bio_ref
            )

            loss = actor_loss + 0.5 * critic_loss + bio_preservation_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        avg_ep_reward = ep_reward / (
            STEPS_PER_UPDATE / 200.0
        )  # Stima normalizzata
        print(
            f"Update {episode:03d}/{MAX_EPISODES} | Reward Medio Reale: {avg_ep_reward:6.1f} | Loss: {loss.item():.4f}"
        )

        if avg_ep_reward > best_reward:
            best_reward = avg_ep_reward
            torch.save(
                {"model_state": model.state_dict(), "body_ids": body_ids},
                "snn_ppo_lunar_winner.pt",
            )

    env.close()
    print("\nTraining Completato. Salvataggio checkpoint in 'snn_ppo_lunar_winner.pt'.")


if __name__ == "__main__":
    train_ppo()
