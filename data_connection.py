from neuprint import Client, fetch_adjacencies, fetch_neurons

token = "9f913383dbd800f5078cdf27745a225f5accfff5661b4aa8b975ac7ab3322e83"
client = Client(
    "https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token
)

neurons, syndist = fetch_neurons("DNge104")
outgoing_edges, neuron_info = fetch_adjacencies("DNge104")
incoming_edges, neuron_info2 = fetch_adjacencies(None, "DNge104")

print("--- Neuroni trovati ---")
print(neurons[["bodyId", "type", "instance"]])
print("\n--- Connessioni in USCITA (verso chi mandano segnali) ---")
print(outgoing_edges.head())
