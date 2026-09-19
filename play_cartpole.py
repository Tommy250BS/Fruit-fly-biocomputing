import time
import gymnasium as gym
import numpy as np
from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

# ==========================================
# 1. CONNESSIONE E MATRICE BIOLOGICA FISSA
# ==========================================
token = "9f913383dbd800f5078cdf27745a225f5accfff5661b4aa8b975ac7ab3322e83"
client = Client(
    "https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token
)

print("Caricamento circuito dal Central Complex...")
neurons, _ = fetch_neurons(NC(rois=["CX"]))
selected_neurons = neurons.head(20)
body_ids = selected_neurons["bodyId"].tolist()

neuron_df, conn_df = fetch_adjacencies(
    NC(bodyId=body_ids), NC(bodyId=body_ids)
)

N = len(body_ids)
W_bio = np.zeros((N, N))
id_to_idx = {b_id: idx for idx, b_id in enumerate(body_ids)}

for _, row in conn_df.iterrows():
    if row["bodyId_pre"] in id_to_idx and row["bodyId_post"] in id_to_idx:
        src = id_to_idx[row["bodyId_pre"]]
        dst = id_to_idx[row["bodyId_post"]]
        W_bio[dst, src] += row["weight"]

if W_bio.max() > 0:
    W_bio = W_bio / W_bio.max()


# ==========================================
# 2. MODELLO NEURONALE (LIF)
# ==========================================
class ReservoirBrain:

    def __init__(self, N_neurons, weight_matrix):
        self.N = N_neurons
        self.W = weight_matrix
        self.v_rest = -70.0
        self.v_thresh = -50.0
        self.v_reset = -75.0
        self.tau = 10.0
        self.reset_state()

    def reset_state(self):
        self.v = np.ones(self.N) * self.v_rest

    def step(self, external_currents):
        activity = np.maximum(0, self.v - self.v_rest)
        synaptic_input = np.dot(self.W, activity)

        dv = (
            -(self.v - self.v_rest) + external_currents + synaptic_input
        ) / self.tau
        self.v += dv

        spikes = self.v >= self.v_thresh
        self.v[spikes] = self.v_reset
        return spikes


# ==========================================
# 3. ADDESTRAMENTO AD ALTA VELOCITÀ
# ==========================================
# Nessun render_mode -> Zero overhead grafico
env = gym.make("CartPole-v1")
brain = ReservoirBrain(N, W_bio)

best_W_out = np.random.normal(0, 1.0, size=N)
best_avg_score = 0

print("\n--- Addestramento Veloce (Massima Velocità di Calcolo) ---")
start_time = time.time()

for episode in range(1, 101):
    if episode == 1:
        W_out = best_W_out.copy()
    else:
        W_out = best_W_out + np.random.normal(0, 0.3, size=N)

    test_scores = []
    for test_run in range(3):
        brain.reset_state()
        state, info = env.reset()
        score = 0

        for t in range(500):
            # Input: 8 canali per le 4 variabili
            ext_currents = np.zeros(N)
            for i in range(4):
                val = state[i]
                if val > 0:
                    ext_currents[i * 2] = val * 25.0
                else:
                    ext_currents[i * 2 + 1] = abs(val) * 25.0

            # Sub-stepping (10 ms)
            spike_counts = np.zeros(N)
            for _ in range(10):
                spikes = brain.step(ext_currents)
                spike_counts += spikes

            # Readout Layer
            neuron_activity = (brain.v - brain.v_rest) + (spike_counts * 10.0)
            motor_signal = np.dot(W_out, neuron_activity)

            action = 1 if motor_signal > 0 else 0
            state, reward, terminated, truncated, info = env.step(action)
            score += 1

            if terminated or truncated:
                break

        test_scores.append(score)

    avg_score = np.mean(test_scores)

    if avg_score > best_avg_score:
        best_avg_score = avg_score
        best_W_out = W_out.copy()
        print(
            f"Tentativo {episode:3d}: *** NUOVO RECORD *** -> {avg_score:.1f} frame! (Singoli: {test_scores})"
        )
    else:
        print(
            f"Tentativo {episode:3d}: Media {avg_score:.1f} frame. (Singoli: {test_scores})"
        )

env.close()

# Salvataggio automatico del modello vincente
np.savez(
    "fly_cartpole_winner.npz",
    W_bio=W_bio,
    W_out=best_W_out,
    body_ids=body_ids,
)

print(f"\nAddestramento completato in {time.time() - start_time:.2f} secondi.")
print("Cervello vincente salvato con successo in 'fly_cartpole_winner.npz'!")