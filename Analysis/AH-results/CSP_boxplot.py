import os, re, pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path fairness_results (Fair-PPO, FEN, SOTO)
BASE_DIR = "ADD_PATH"
FAIRPPO_DIR = os.path.join(BASE_DIR, "ADD_PATH")
FEN_DIR     = os.path.join(BASE_DIR, "ADD_PATH")
SOTO_DIR    = os.path.join(BASE_DIR, "ADD_PATH")

output_dir  = "ADD_PATH"
os.makedirs(output_dir, exist_ok=True)
# ----------------------------------------------------------------

# **Desired x-axis order**
ALGORITHMS = [
    "fair-PPO",
    "vanilla-PPO",
    "FEN",
    "SOTO"
]

base_colors: Dict[str, str] = {
    "fair-PPO" : "#7570b3", 
    "vanilla-PPO": "#66b3ff", 
    "FEN" : "#1b9e77",  
    "SOTO" : "#d95f02",
}

def extract_episode_lists(blob):
    """
    Extracts raw reward lists per episode for the four main subgroups
    """
    
    cumul = blob.get("cumulative_rewards", {})
    out = {g: list(cumul.get(g, [])) for g in (
        "non_sensitive_red", "non_sensitive_blue", "sensitive_red", "sensitive_blue")}
    
    n_eps = max((len(v) for v in out.values()), default=0)
    for g in out:
        out[g].extend([0]*(n_eps-len(out[g])))
    return out

def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if x.sum() == 0: return 0.0
    return (2 * (np.arange(1, n + 1) * x).sum()) / (n * x.sum()) - (n + 1) / n

def jain(x):
    x = np.asarray(x, float); s = x.sum()
    if s == 0: return 0.0
    return s**2 / (len(x) * (x**2).sum())

def nash_norm(x):
    x = np.asarray(x, float) + 1e-10
    return (x.prod()**(1/len(x))) / x.mean()

def per_episode_metrics_for_group(epl, group):
    """
    Calculates metrics for a specific group (Red or Blue)
    """
    
    disp, gi, jf, nn = [], [], [], []
    
    if group == 'Red':
        r_ns_list, r_s_list = epl["non_sensitive_red"], epl["sensitive_red"]
    elif group == 'Blue':
        r_ns_list, r_s_list = epl["non_sensitive_blue"], epl["sensitive_blue"]
    else:
        return [], [], [], []

    for r_ns, r_s in zip(r_ns_list, r_s_list):
        rewards = np.array([r_ns, r_s])
        disp.append(r_ns - r_s)
        gi.append(gini(rewards))
        jf.append(jain(rewards))
        nn.append(nash_norm(rewards))
        
    return disp, gi, jf, nn

def load_folder(folder, regex, cast):
    rx = re.compile(regex); out = {}
    if not os.path.isdir(folder):
        print(f"Directory not found, skipping: {folder}")
        return out
    for f in os.listdir(folder):
        m = rx.match(f)
        if m:
            try:
                with open(os.path.join(folder, f), "rb") as fp:
                    out[cast(*m.groups())] = pickle.load(fp)
            except (pickle.UnpicklingError, EOFError):
                 print(f"Could not load corrupted file: {os.path.join(folder, f)}")
    return out

# LOADERS
ALL_FP = load_folder(FAIRPPO_DIR, r"fairness_results_alpha=([0-9.]+)_beta=([0-9.]+)\.pkl", lambda a, b: (float(a), float(b)))
if not any(abs(k[0])<1e-9 and abs(k[1])<1e-9 for k in ALL_FP):
    vanilla_data = load_folder(FAIRPPO_DIR, r"fairness_results\.pkl", str)
    if vanilla_data:
        ALL_FP[(0.0, 0.0)] = next(iter(vanilla_data.values()))


LOADERS = {
    "vanilla-PPO": lambda: {k: v for k, v in ALL_FP.items() if abs(k[0])<1e-9 and abs(k[1])<1e-9},
    "fair-PPO": lambda: {k: v for k, v in ALL_FP.items() if not(abs(k[0])<1e-9 and abs(k[1])<1e-9)},
    "FEN": lambda: load_folder(FEN_DIR , r"fairness_results\.pkl", str),
    "SOTO": lambda: load_folder(SOTO_DIR, r"fairness_results_alpha=(.+)\.pkl", str),
}


# Data processing
rows = []
for algo in ALGORITHMS:
    for knob, blob in LOADERS[algo]().items():
        epl = extract_episode_lists(blob)
        knob_str = "" if algo in ("vanilla-PPO", "FEN") else (
            f"α-fair={knob}" if algo=="SOTO" else f"α={knob[0]},β={knob[1]}")
            
        # Analysis for Red Preference Group
        disp_r, gi_r, jf_r, nn_r = per_episode_metrics_for_group(epl, 'Red')
        for a, b, c, d in zip(disp_r, gi_r, jf_r, nn_r):
            rows.append(dict(Group="Red Preference", Algorithm=algo, Knob=knob_str, Metric="Disparity", Value=a))
            rows.append(dict(Group="Red Preference", Algorithm=algo, Knob=knob_str, Metric="Gini", Value=b))
            rows.append(dict(Group="Red Preference", Algorithm=algo, Knob=knob_str, Metric="JFI", Value=c))
            rows.append(dict(Group="Red Preference", Algorithm=algo, Knob=knob_str, Metric="NNSW", Value=d))

        # Analysis for Blue Preference Group
        disp_b, gi_b, jf_b, nn_b = per_episode_metrics_for_group(epl, 'Blue')
        for a, b, c, d in zip(disp_b, gi_b, jf_b, nn_b):
            rows.append(dict(Group="Blue Preference", Algorithm=algo, Knob=knob_str, Metric="Disparity", Value=a))
            rows.append(dict(Group="Blue Preference", Algorithm=algo, Knob=knob_str, Metric="Gini", Value=b))
            rows.append(dict(Group="Blue Preference", Algorithm=algo, Knob=knob_str, Metric="JFI", Value=c))
            rows.append(dict(Group="Blue Preference", Algorithm=algo, Knob=knob_str, Metric="NNSW", Value=d))
            
long_df = pd.DataFrame(rows)

# Normalise Disparity
max_disp = long_df.loc[long_df.Metric == "Disparity", "Value"].abs().max()
if max_disp > 0:
    long_df.loc[long_df.Metric == "Disparity", "Value"] /= max_disp
long_df.loc[long_df.Metric == "Disparity", "Metric"] = "Disparity_scaled"


for group_name in ["Red Preference", "Blue Preference"]:
    print(f"Processing results for: {group_name}")
    
    group_df = long_df[long_df.Group == group_name]
    
    # 1. Find best knob for this group based on lowest disparity
    best = {}
    for algo in ALGORITHMS:
        sub = group_df[(group_df.Algorithm == algo) & (group_df.Metric == "Disparity_scaled")]
        if not sub.empty:
            knob_scores = sub.groupby("Knob").Value.apply(lambda s: s.abs().mean())
            if not knob_scores.empty:
                best[algo] = knob_scores.idxmin()
    
    best_df = group_df[group_df.apply(lambda r: r.Knob == best.get(r.Algorithm, ""), axis=1)]

    if best_df.empty:
        print(f"No data found for {group_name}. Skipping.")
        continue

    # 2. Plotting
    metric_names = {
        "Disparity_scaled": "Disparity",
        "Gini": "Gini Index",
        "NNSW": "Normalised Nash SW"
    }
    
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    group_id = group_name.split(' ')[0]
    fig.suptitle(f"Best-Disparity Configuration per Algorithm ({group_id} Group)", fontsize=16, fontweight="bold")
    
    palette = {a: base_colors.get(a, "#000000") for a in ALGORITHMS}

    for ax, (mkey, title) in zip(axes, metric_names.items()):
        sns.boxplot(
            ax=ax,
            data=best_df[best_df.Metric == mkey],
            x="Algorithm", y="Value",
            order=ALGORITHMS, hue="Algorithm", palette=palette, legend=False,
            width=0.8, showfliers=False, whis=99
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(title, fontsize=11)
        ax.tick_params(axis="x", labelsize=10, rotation=15)
        ax.yaxis.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, f"metrics_boxplots_best_{group_id}.svg")
    plt.savefig(plot_path)
    plt.close()
    print(f"Box-plot saved {plot_path}")
    
    # 3. LaTeX summary
    summary = (
        best_df.groupby(["Algorithm", "Metric"]).Value.mean()
        .unstack("Metric")
        .reset_index()
        .rename(columns={
            "Disparity_scaled": "Disparity (↓)",
            "Gini": "Gini (↓)",
            "JFI": "JFI (↑)",
            "NNSW": "NNSW (↑)"
        })
    )
    final_cols = ["Algorithm", "Disparity (↓)", "Gini (↓)", "JFI (↑)", "NNSW (↑)"]
    summary = summary[[col for col in final_cols if col in summary.columns]]

    summary["_o"] = summary.Algorithm.apply(lambda x: ALGORITHMS.index(x) if x in ALGORITHMS else -1)
    summary = summary.sort_values("_o").drop(columns="_o")

    tex = summary.to_latex(
        index=False, float_format="%.3f",
        caption=f"Mean metrics (best disparity for {group_id} group) per algorithm",
        label=f"tab:best_configs_metrics_{group_id.lower()}",
        column_format="lcccc", escape=False
    )
    tex_path = os.path.join(output_dir, f"best_configs_metrics_table_{group_id.lower()}.tex")
    with open(tex_path, "w") as fp:
        fp.write(tex)
