import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Administrator\Downloads\csv (7).csv")

# Auto-detect epoch and loss columns (case-insensitive)
cols = {c.lower(): c for c in df.columns}
epoch_col = next((cols[k] for k in cols if "epoch" in k or "step" in k), df.columns[1])
loss_col  = next((cols[k] for k in cols if "loss" in k), df.columns[2])

loss = df[loss_col]
v_max    = loss.max()
v_min    = loss.min()
v_median = loss.median()
v_mean   = loss.mean()

plt.figure(figsize=(8, 5))
plt.plot(df[epoch_col], loss, label="Loss")
plt.axhline(v_max,    color="red",    linestyle="--", linewidth=1, label=f"Max    {v_max:.4f}")
plt.axhline(v_min,    color="green",  linestyle="--", linewidth=1, label=f"Min    {v_min:.4f}")
plt.axhline(v_median, color="orange", linestyle="--", linewidth=1, label=f"Median {v_median:.4f}")
plt.axhline(v_mean,   color="blue",   linestyle=":",  linewidth=1, label=f"Mean   {v_mean:.4f}")
plt.xlabel(epoch_col)
plt.ylabel(loss_col)
plt.title("Loss vs Epoch")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
