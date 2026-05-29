import numpy as np
import matplotlib.pyplot as plt
import os

# Read the dat file — skip the header line (starts with #)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, "out", "steadystate_du_Linf.dat")

data      = np.loadtxt(data_path, comments="#")
timesteps = data[:, 0]
time      = data[:, 1]
du_linf   = data[:, 2]

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(time, du_linf, color="blue", linewidth=1.0)
ax.set_xlabel("sim time (s)")
ax.set_ylabel(r"$\|du\|_\infty$")
ax.set_title("Steady State Convergence")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()

# Save
out_path = os.path.join(script_dir, "convergence.png")
plt.savefig(out_path, dpi=150)
print(f"Saved to: {out_path}")
plt.show()