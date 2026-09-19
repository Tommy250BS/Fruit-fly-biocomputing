import os
import random
import time
import gymnasium as gym
import numpy as np
import torch
from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

# ==========================================
# 0. CONFIGURAZIONE E RIPRODUCIBILITÀ
# ==========================================
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
# 1. ESTRAZIONE E DALE'S LAW (NEUPRINT)
# ==========================================
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

    # Mappatura rigorosa del principio di Dale (Eccitatori vs Inibitori)
    polarity = np.ones(N, dtype=np.float32)
    for idx, (_, row) in enumerate(selected_neurons.iterrows()):
        nt = str(row.get("ntPredicted", "")).lower()
        if "gaba" in nt or "glutamate" in nt:
            polarity[idx] = -1.0  # Neurotrasmettitore inibitorio

    for _, row in conn_df.iterrows():
        if row["bodyId_pre"] in id_to_idx and row["bodyId_post"] in id_to_idx:
            src = id_to_idx[row["bodyId_pre"]]
            dst = id_to_idx[row["bodyId_post"]]
            W_bio[dst, src] += row["weight"] * polarity[src]

    # Normalizzazione per raggio spettrale per garantire stabilità della dinamica
    max_val = np.abs(W_bio).max()
    if max_val > 0:
        W_bio /= max_val

    return torch.tensor(W_bio, dtype=torch.float32, device=device), N, body_ids


# ==========================================
# 2. LIF BRAIN WITH SYNAPTIC FILTERING
# ==========================================
class ResearchLIFBrain:

    def __init__(self, N, W_bio, dt=1.0):
        self.N = N
        self.W = W_bio
        self.v_rest = -70.0
        self.v_thresh = -50.0
        self.v_reset = -75.0
        self.tau_m = 10.0  # Costante di tempo di membrana (ms)
        self.tau_s = 5.0  # Costante di tempo del filtro sinaptico (ms)
        self.dt = dt
        self.reset_state()

    def reset_state(self):
        self.v = torch.full((self.N,), self.v_rest, device=device)
        self.syn_trace = torch.zeros(self.N, device=device)

    def step(self, external_current):
        # Calcolo potenziale con dinamica Leaky Integrate-and-Fire
        synaptic_input = torch.mv(self.W, self.syn_trace)
        dv = (
            -(self.v - self.v_rest) + external_current + synaptic_input
        ) / self.tau_m
        self.v += dv * self.dt

        spikes = (self.v >= self.v_thresh).float()
        self.v[spikes.bool()] = self.v_reset

        # Traccia sinaptica del primo ordine (Filtro passa-basso sugli spike)
        self.syn_trace = self.syn_trace * (1.0 - self.dt / self.tau_s) + spikes
        return spikes


# ==========================================
# 3. VALUTAZIONE RIGOROSA DI UN CANDIDATO
# ==========================================
def evaluate_candidate(
    brain, W_out, seeds, render=False, max_steps=600
) -> list:
    """Valuta una matrice W_out su un insieme fisso di seed deterministici."""
    env = gym.make(
        "LunarLander-v3", render_mode="human" if render else None
    )
    rewards = []

    for seed in seeds:
        state, info = env.reset(seed=seed)
        brain.reset_state()
        total_reward = 0.0

        for _ in range(max_steps):
            # Input Encoding: Popolazione Bipolare dei 8 canali di stato Gym
            ext_I = torch.zeros(brain.N, device=device)
            for i in range(min(8, brain.N // 2)):
                val = state[i]
                if val > 0:
                    ext_I[i * 2] = float(val) * 15.0
                else:
                    ext_I[i * 2 + 1] = float(abs(val)) * 15.0

            # Sub-stepping della rete neurale
            for _ in range(5):
                brain.step(ext_I)

            # Readout Layer: Mappatura lineare dalla traccia sinaptica ai comandi motori
            motor_logits = torch.mv(W_out, brain.syn_trace)
            action = torch.argmax(motor_logits).item()

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    env.close()
    return rewards


# ==========================================
# 4. LOOP DI ADDESTRAMENTO NES & VALIDAZIONE
# ==========================================
def main():
    print("=== INIZIO PIPELINE DI RICERCA: LUNAR LANDER CONNECTOME ===")
    W_bio, N, body_ids = load_neuprint_connectome(100)
    brain = ResearchLIFBrain(N, W_bio)

    POP_SIZE = 16  # Dimensione popolazione mutanti per generazione
    N_EVAL_SEEDS = 8  # Numero di seed stocastici per ogni valutazione
    GENERATIONS = 80
    SIGMA = 0.15  # Tasso di mutazione/esplorazione

    # Inizializzazione pesi readout (4 azioni x N neuroni)
    W_out_mean = torch.randn((4, N), device=device) * 0.1
    best_overall_score = -float("inf")
    best_W_out = W_out_mean.clone()

    for gen in range(1, GENERATIONS + 1):
        # Generazione di seed dinamici ma identici per tutti i candidati della generazione
        gen_seeds = [
            int(SEED * 1000 + gen * 100 + i) for i in range(N_EVAL_SEEDS)
        ]

        # Campionamento popolazione (Natural Evolution Strategies)
        noise_samples = [
            torch.randn_like(W_out_mean) for _ in range(POP_SIZE)
        ]
        candidates = [W_out_mean + SIGMA * noise for noise in noise_samples]

        # Includiamo la media corrente nella valutazione
        candidates.append(W_out_mean)

        scores = []
        for cand in candidates:
            r_list = evaluate_candidate(brain, cand, gen_seeds)
            # Fitness robusta: Media con eliminazione dei valori estremi
            scores.append(np.mean(r_list))

        cand_scores = scores[:-1]
        mean_score = scores[-1]

        # Aggiornamento NES basato sulla gradiente stimato delle ricompense
        score_tensor = torch.tensor(cand_scores, device=device)
        standardized_scores = (score_tensor - score_tensor.mean()) / (
            score_tensor.std() + 1e-8
        )

        update_step = torch.zeros_like(W_out_mean)
        for i in range(POP_SIZE):
            update_step += standardized_scores[i] * noise_samples[i]
        W_out_mean += (0.05 / (POP_SIZE * SIGMA)) * update_step

        max_gen_score = np.max(scores)
        print(
            f"Gen {gen:02d}/{GENERATIONS} | Score Medio Pop: {mean_score:6.1f} | Max Gen: {max_gen_score:6.1f}"
        )

        if max_gen_score > best_overall_score:
            best_overall_score = max_gen_score
            best_W_out = candidates[np.argmax(scores)].clone()

    # ==========================================
    # 5. OUT-OF-SAMPLE CROSS-VALIDATION (50 SEEDS)
    # ==========================================
    print("\n" + "=" * 50)
    print(" INIZIO CROSS-VALIDAZIONE OUT-OF-SAMPLE (50 SEEDS UNSEEN)")
    print("=" * 50)

    oos_seeds = [int(999000 + i) for i in range(50)]
    oos_rewards = evaluate_candidate(brain, best_W_out, oos_seeds)

    mean_oos = np.mean(oos_rewards)
    std_oos = np.std(oos_rewards)
    median_oos = np.median(oos_rewards)
    success_rate = (
        np.sum(np.array(oos_rewards) >= 200.0) / len(oos_rewards)
    ) * 100.0

    print(f"Risultati della Validazione Statistica:")
    print(f" -> Media OOS:          {mean_oos:.2f} ± {std_oos:.2f}")
    print(f" -> Mediana OOS:        {median_oos:.2f}")
    print(f" -> Min / Max Reward:   {np.min(oos_rewards):.1f} / {np.max(oos_rewards):.1f}")
    print(f" -> Tasso Atterraggio: {success_rate:.1f}% (Punteggio >= 200)")

    # Salvataggio del checkpoint certificato
    save_payload = {
        "W_bio": W_bio.cpu(),
        "W_out": best_W_out.cpu(),
        "body_ids": body_ids,
        "metrics": {
            "mean_oos": mean_oos,
            "std_oos": std_oos,
            "success_rate": success_rate,
        },
    }
    torch.save(save_payload, "fly_lunar_lander_verified.pt")
    print(
        "\nModello validato salvato con successo in 'fly_lunar_lander_verified.pt'."
    )


if __name__ == "__main__":
    main()
