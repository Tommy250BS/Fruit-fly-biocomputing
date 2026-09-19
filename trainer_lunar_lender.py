import time
import gymnasium as gym
import numpy as np
import torch
from neuprint import Client, fetch_adjacencies, fetch_neurons, NeuronCriteria as NC

# Seleziona il dispositivo di calcolo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")

# ==========================================
# 1. ARCHITETTURA BIOLOGICA DA NEUPRINT
# ==========================================
token = "9f913383dbd800f5078cdf27745a225f5accfff5661b4aa8b975ac7ab3322e83"
client = Client(
    "https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token
)

print(
    "Estrazione circuito di navigazione spaziale (FB, EB) da NeuPrint..."
)
neurons, _ = fetch_neurons(NC(rois=["EB", "FB"]))

# Prendiamo i primi 100 neuroni del Central Complex
selected_neurons = neurons.head(100)
body_ids = selected_neurons["bodyId"].tolist()

neuron_df, conn_df = fetch_adjacencies(
    NC(bodyId=body_ids), NC(bodyId=body_ids)
)

N = len(body_ids)
W_bio_np = np.zeros((N, N))
id_to_idx = {b_id: idx for idx, b_id in enumerate(body_ids)}

# Mappatura della polarità dei neurotrasmettitori (GABA/Glutammato = -1, Acetilcolina = +1)
polarity_map = {}
for _, row in selected_neurons.iterrows():
    b_id = row["bodyId"]
    nt = str(row.get("ntPredicted", "")).lower()
    if "gaba" in nt or "glutamate" in nt:
        polarity_map[b_id] = -1.0  # Inibitorio
    else:
        polarity_map[b_id] = 1.0  # Eccitatorio (Default ChAT / Acetilcolina)

for _, row in conn_df.iterrows():
    if row["bodyId_pre"] in id_to_idx and row["bodyId_post"] in id_to_idx:
        src = id_to_idx[row["bodyId_pre"]]
        dst = id_to_idx[row["bodyId_post"]]
        src_id = row["bodyId_pre"]

        sign = polarity_map.get(src_id, 1.0)
        W_bio_np[dst, src] += row["weight"] * sign

# Normalizzazione dei pesi della matrice
max_weight = np.abs(W_bio_np).max()
if max_weight > 0:
    W_bio_np = W_bio_np / max_weight

# Conversione in Tensore PyTorch
W_bio_tensor = torch.tensor(W_bio_np, dtype=torch.float32, device=device)


# ==========================================
# 2. CERVELLO BIOLOGICO VETTORIZZATO (PyTorch)
# ==========================================
class PyTorchLIFBrain:

    def __init__(self, N_neurons, weight_matrix):
        self.N = N_neurons
        self.W = weight_matrix
        self.v_rest = -70.0
        self.v_thresh = -50.0
        self.v_reset = -75.0
        self.tau = 10.0
        self.reset_state()

    def reset_state(self):
        self.v = torch.full(
            (self.N,), self.v_rest, dtype=torch.float32, device=device
        )

    def step(self, external_currents):
        activity = torch.clamp(self.v - self.v_rest, min=0.0)
        synaptic_input = torch.mv(self.W, activity)

        dv = (
            -(self.v - self.v_rest) + external_currents + synaptic_input
        ) / self.tau
        self.v += dv

        spikes = self.v >= self.v_thresh
        self.v[spikes] = self.v_reset
        return spikes.float()


# ==========================================
# 3. ADDESTRAMENTO PER LUNAR LANDER
# ==========================================
# Generazione ambiente (LunarLander ha 8 input e 4 azioni motorie)
try:
    env = gym.make("LunarLander-v3")
except:
    env = gym.make("LunarLander-v2")

brain = PyTorchLIFBrain(N, W_bio_tensor)

# Matrice di Readout: connette i N neuroni ai 4 comandi motori (Dimensione: 4 x N)
best_W_out = torch.randn((4, N), dtype=torch.float32, device=device) * 0.5
best_avg_reward = -999.0

print(
    f"\n--- Inizio Addestramento PyTorch su LunarLander ({N} Neuroni) ---"
)
start_time = time.time()

for episode in range(1, 151):
    if episode == 1:
        W_out = best_W_out.clone()
    else:
        # Mutazione adattiva dei pesi motori
        mutation_scale = max(0.05, 0.4 - (best_avg_reward / 300.0))
        W_out = best_W_out + torch.randn_like(best_W_out) * mutation_scale

    test_rewards = []
    for test_run in range(3):
        brain.reset_state()
        state, info = env.reset()
        total_reward = 0.0

        for t in range(600):
            # Encoding degli 8 canali dello stato in 16 canali elettrici (Positivo / Negativo)
            ext_currents = torch.zeros(N, dtype=torch.float32, device=device)
            for i in range(min(8, N // 2)):
                val = state[i]
                if val > 0:
                    ext_currents[i * 2] = float(val) * 20.0
                else:
                    ext_currents[i * 2 + 1] = float(abs(val)) * 20.0

            # Sub-stepping della rete neurale (10 ms di dinamica di membrana)
            spike_counts = torch.zeros(N, dtype=torch.float32, device=device)
            for _ in range(10):
                spikes = brain.step(ext_currents)
                spike_counts += spikes

            # Decoding del Readout Layer: Calcolo della spinta sui 4 motori
            neuron_activity = (brain.v - brain.v_rest) + (spike_counts * 10.0)
            motor_signals = torch.mv(W_out, neuron_activity)

            # Scelta dell'azione motoria a potenziale più alto
            action = torch.argmax(motor_signals).item()

            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        test_rewards.append(total_reward)

    avg_reward = np.mean(test_rewards)

    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        best_W_out = W_out.clone()
        print(
            f"Episodio {episode:3d}: *** NUOVO RECORD *** -> Ricompensa Media: {avg_reward:.1f} (Singoli: {[round(r,1) for r in test_rewards]})"
        )
    else:
        print(
            f"Episodio {episode:3d}: Ricompensa Media: {avg_reward:.1f} (Singoli: {[round(r,1) for r in test_rewards]})"
        )

env.close()

# Salvataggio del modello PyTorch e della matrice
torch.save(
    {
        "W_bio": W_bio_tensor.cpu(),
        "W_out": best_W_out.cpu(),
        "body_ids": body_ids,
        "N": N,
    },
    "fly_lunar_lander_winner.pt",
)

print(
    f"\nAddestramento completato in {time.time() - start_time:.2f} secondi."
)
print(
    "Cervello vincente salvato con successo in 'fly_lunar_lander_winner.pt'!"
)
