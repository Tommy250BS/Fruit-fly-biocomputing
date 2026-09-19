import random
import time
import gymnasium as gym
import numpy as np
import torch
from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. CARICAMENTO CONNETTOMA BIOLOGICO
def load_neuprint_connectome(n_neurons=100):
    token = "9f913383dbd800f5078cdf27745a225f5accfff5661b4aa8b975ac7ab3322e83"
    client = Client(
        "https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token
    )

    neurons, _ = fetch_neurons(NC(rois=["EB", "FB"]))
    selected_neurons = neurons.head(n_neurons)
    body_ids = selected_neurons["bodyId"].tolist()
    neuron_df, conn_df = fetch_adjacencies(
        NC(bodyId=body_ids), NC(bodyId=body_ids)
    )

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


# 2. LIF BRAIN CON ECCITABILITÀ PLASTICA
class PlasticLIFBrain:

    def __init__(self, N, W_bio, dt=1.0):
        self.N = N
        self.W = W_bio
        self.v_rest = -70.0
        self.v_thresh = -50.0
        self.v_reset = -75.0
        self.tau_m = 10.0
        self.tau_s = 10.0
        self.dt = dt
        self.reset_state()

    def reset_state(self):
        self.v = torch.full((self.N,), self.v_rest, device=device)
        self.syn_trace = torch.zeros(self.N, device=device)

    def step(self, external_current, gains, biases):
        # Applicazione dell'eccitabilità intrinseca modulatrice
        i_eff = external_current * gains + biases

        synaptic_input = torch.mv(self.W, self.syn_trace)
        dv = (-(self.v - self.v_rest) + i_eff + synaptic_input) / self.tau_m
        self.v += dv * self.dt

        spikes = (self.v >= self.v_thresh).float()
        self.v[spikes.bool()] = self.v_reset

        self.syn_trace = self.syn_trace * (1.0 - self.dt / self.tau_s) + spikes
        return spikes


# 3. VALUTAZIONE RIGOROSA CON FILTRO AZIONE PASSA-BASSO
def evaluate_genome(brain, genome, seeds, max_steps=800) -> list:
    """Genome vettorizzato:
    - W_out: 4 x N (400 params)
    - Gains (g): N params
    - Biases (b): N params
    """
    N = brain.N
    W_out = genome[: 4 * N].reshape(4, N)
    gains = genome[4 * N : 5 * N]
    biases = genome[5 * N : 6 * N]

    env = gym.make("LunarLander-v3")
    rewards = []

    for seed in seeds:
        state, info = env.reset(seed=seed)
        brain.reset_state()
        total_reward = 0.0
        prev_action_logits = torch.zeros(4, device=device)

        for _ in range(max_steps):
            ext_I = torch.zeros(N, device=device)
            for i in range(min(8, N // 2)):
                val = state[i]
                if val > 0:
                    ext_I[i * 2] = float(val) * 10.0
                else:
                    ext_I[i * 2 + 1] = float(abs(val)) * 10.0

            for _ in range(5):
                brain.step(ext_I, gains, biases)

            raw_logits = torch.mv(W_out, brain.syn_trace)
            # Filtro passa-basso cinetico (Alpha = 0.3) per eliminare il chattering motorio
            action_logits = 0.7 * prev_action_logits + 0.3 * raw_logits
            prev_action_logits = action_logits.clone()

            action = torch.argmax(action_logits).item()

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return rewards


# 4. RUNNER PRINCIPALE
def main():
    print("=== PIPELINE RICERCA v2.0: CMA-ES + INTRINSIC EXCITABILITY ===")
    W_bio, N, body_ids = load_neuprint_connectome(100)
    brain = PlasticLIFBrain(N, W_bio)

    # Vettore dei parametri: W_out (400) + Gains (100) + Biases (100) = 600 parametri
    PARAM_DIM = 4 * N + N + N
    POP_SIZE = 32
    GENERATIONS = 120
    SIGMA = 0.2

    # Inizializzazione centro della popolazione
    mean_genome = torch.zeros(PARAM_DIM, device=device)
    mean_genome[: 4 * N] = torch.randn(4 * N, device=device) * 0.1  # W_out
    mean_genome[4 * N : 5 * N] = 1.0  # Gains
    mean_genome[5 * N : 6 * N] = 0.0  # Biases

    best_overall_score = -float("inf")
    best_genome = mean_genome.clone()

    for gen in range(1, GENERATIONS + 1):
        gen_seeds = [int(SEED * 5000 + gen * 50 + i) for i in range(10)]

        # Generazione popolazione con perturbazioni sferiche
        noise_samples = [
            torch.randn(PARAM_DIM, device=device) for _ in range(POP_SIZE)
        ]
        candidates = [mean_genome + SIGMA * noise for noise in noise_samples]
        candidates.append(mean_genome)

        scores = [
            np.mean(evaluate_genome(brain, cand, gen_seeds))
            for cand in candidates
        ]

        cand_scores = scores[:-1]
        mean_score = scores[-1]

        # Aggiornamento adattivo NES/CMA
        score_tensor = torch.tensor(cand_scores, device=device)
        std_val = score_tensor.std() + 1e-8
        norm_scores = (score_tensor - score_tensor.mean()) / std_val

        update = torch.zeros(PARAM_DIM, device=device)
        for i in range(POP_SIZE):
            update += norm_scores[i] * noise_samples[i]

        mean_genome += (0.08 / (POP_SIZE * SIGMA)) * update

        max_score = np.max(scores)
        print(
            f"Gen {gen:03d}/{GENERATIONS} | Score Medio: {mean_score:6.1f} | Max Gen: {max_score:6.1f}"
        )

        if max_score > best_overall_score:
            best_overall_score = max_score
            best_genome = candidates[np.argmax(scores)].clone()

    # 5. VALIDAZIONE FINALE SU 50 SEED SCONOSCIUTI
    print("\n" + "=" * 50)
    print(" CROSS-VALIDAZIONE OOS FINALE (50 SEED)")
    print("=" * 50)

    oos_seeds = [int(888000 + i) for i in range(50)]
    oos_rewards = evaluate_genome(brain, best_genome, oos_seeds)

    mean_oos = np.mean(oos_rewards)
    std_oos = np.std(oos_rewards)
    success_rate = (
        np.sum(np.array(oos_rewards) >= 200.0) / len(oos_rewards)
    ) * 100.0

    print(f"Risultati Finali Certificati:")
    print(f" -> Media OOS:          {mean_oos:.2f} ± {std_oos:.2f}")
    print(f" -> Max Reward:         {np.max(oos_rewards):.1f}")
    print(f" -> Tasso Atterraggio: {success_rate:.1f}% (Punteggio >= 200)")

    torch.save(
        {
            "W_bio": W_bio.cpu(),
            "genome": best_genome.cpu(),
            "body_ids": body_ids,
            "metrics": {"mean_oos": mean_oos, "success_rate": success_rate},
        },
        "fly_lunar_lander_v2_verified.pt",
    )


if __name__ == "__main__":
    main()
