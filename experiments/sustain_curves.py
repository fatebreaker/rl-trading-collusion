"""Most collusive intensity sustainable by trigger strategies, by discount
factor, for the standard Kyle market (xi = 0) and with information-insensitive
investors (xi = 50, 500). Writes results/theory/sustain.json."""

import json
import math

import numpy as np

from kylecollusion.sustain import max_sustainable, max_sustainable_any

CASES = {"xi0": dict(xi=0.0, sigma_u=1.0, n_values=5),
         "xi50": dict(xi=50.0, sigma_u=0.1, n_values=10),
         "xi500": dict(xi=500.0, sigma_u=0.1, n_values=10)}
deltas = [round(x, 3) for x in np.linspace(0.05, 0.95, 19)] + [0.99]
out = {"deltas": deltas}
for name, c in CASES.items():
    out[name] = {
        "grim": [max_sustainable(d, math.inf, **c)["delta_beta"] for d in deltas],
        "T1": [max_sustainable(d, 1, **c)["delta_beta"] for d in deltas],
        "any": [max_sustainable_any(d, **c)["delta_beta"] for d in deltas],
    }
    print(name, {k: [round(x, 2) for x in v] for k, v in out[name].items()}, flush=True)
json.dump(out, open("results/theory/sustain.json", "w"), indent=1)
