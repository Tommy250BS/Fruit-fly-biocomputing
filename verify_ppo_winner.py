import gymnasium as gym
import numpy as np
import torch
from surrogate_lunar_ppo import SpikingActorCritic, load_neuprint_connectome

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento Modello
W_bio, N, body_ids = load_neuprint_connectome(100)
model = SpikingActorCritic(N, W_bio).to(device)

checkpoint = torch.load("snn_ppo_lunar_winner.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 1. CROSS-VALIDAZIONE OUT-OF-SAMPLE (50 SEEDS)
print("=== INIZIO CROSS-VALIDAZIONE STATISTICA OOS (50 SEEDS) ===")
env_eval = gym.make("LunarLander-v3")
oos_rewards = []

for seed in range(50):
    obs, _ = env_eval.reset(seed=9000 + seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    v = torch.full((N,), model.v_rest, device=device)
    syn_trace = torch.zeros(N, device=device)
    total_reward = 0.0

    for step in range(800):
        with torch.no_grad():
            logits, _, v, syn_trace = model.forward_step(obs, v, syn_trace)
            action = torch.argmax(logits).item()

        next_obs, reward, terminated, truncated, _ = env_eval.step(action)
        total_reward += reward
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        if terminated or truncated:
            break

    oos_rewards.append(total_reward)

env_eval.close()

mean_reward = np.mean(oos_rewards)
std_reward = np.std(oos_rewards)
success_rate = (
    np.sum(np.array(oos_rewards) >= 200.0) / len(oos_rewards)
) * 100.0

print(f"Risultati Validazione OOS:")
print(f" -> Media Punteggio:   {mean_reward:.2f} ± {std_reward:.2f}")
print(f" -> Max Punteggio:     {np.max(oos_rewards):.1f}")
print(f" -> Tasso Atterraggio: {success_rate:.1f}% (Punteggio >= 200)")

# 2. RENDERING GRAFICO
env_render = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env_render.reset(seed=42)
obs = torch.tensor(obs, dtype=torch.float32, device=device)

v = torch.full((N,), model.v_rest, device=device)
syn_trace = torch.zeros(N, device=device)
tot_r = 0.0

for _ in range(800):
    env_render.render()
    with torch.no_grad():
        logits, _, v, syn_trace = model.forward_step(obs, v, syn_trace)
        action = torch.argmax(logits).item()

    next_obs, reward, terminated, truncated, _ = env_render.step(action)
    tot_r += reward
    obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    if terminated or truncated:
        print(f"Volo di dimostrazione completato con punteggio: {tot_r:.2f}")
        break

env_render.close()
