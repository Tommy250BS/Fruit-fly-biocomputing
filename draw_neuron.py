import matplotlib.pyplot as plt
import navis
import navis.interfaces.neuprint as neu
from neuprint import Client

token = "9f913383dbd800f5078cdf27745a225f5accfff5661b4aa8b975ac7ab3322e83"
client = Client(
    "https://neuprint.janelia.org", dataset="male-cns:v1.0", token=token
)

skels = neu.fetch_skeletons(neu.NeuronCriteria(type="DNge104"), client=client)

fig, ax = navis.plot2d(skels, view=("z", "x"), radius=True)
plt.show()
