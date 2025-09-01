#!/usr/bin/env python3

import os, re, pickle, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Path fairness_results (Fair-PPO, FEN, SOTO)
FAIRPPO_DIR = "ADD_PATH"
FEN_DIR     = "ADD_PATH"
SOTO_DIR    = "ADD_PATH"

#output dir
OUT_DIR = "ADD_PATH"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="white", context="talk",
              rc={"font.family": "serif", "axes.linewidth": 1.1})

ALGO_COLOURS = {
    # Plot A Colors
    "Fair-PPO (0.25, 0.0)": "#f7d4d6",
    "Fair-PPO (0.25, 0.25)": "#b36a8f",
    "Fair-PPO (0.75, 0.25)": "#20021B",
    # Plot B Colors
    "Vanilla-PPO": "#66b3ff",
    "FEN": "#1b9e77",
    "SOTO": "#d95f02",
}


def _load_folder(folder, pattern, cast_key):
    rx, out = re.compile(pattern), {}
    for fname in os.listdir(folder):
        m = rx.match(fname)
        if not m: continue
        with open(os.path.join(folder, fname), "rb") as fp:
            out[cast_key(*m.groups())] = pickle.load(fp)
    return out

# load selected algorithms based on the hyperparameters (example)
LOADERS = {
    "Fair-PPO (0.25, 0.0)": lambda: _load_folder(FAIRPPO_DIR, r"fairness_results_alpha=(0.25)_beta=(0.0)\.pkl",
                                        lambda a,b: (float(a), float(b))),
    "Fair-PPO (0.25, 0.25)": lambda: _load_folder(FAIRPPO_DIR, r"fairness_results_alpha=(0.25)_beta=(0.25)\.pkl",
                                    lambda a,b: (float(a), float(b))),
    "Fair-PPO (0.75, 0.25)": lambda: _load_folder(FAIRPPO_DIR, r"fairness_results_alpha=(0.75)_beta=(0.25)\.pkl",
                                        lambda a,b: (float(a), float(b))),
    "Vanilla-PPO": lambda: _load_folder(FAIRPPO_DIR, r"fairness_results_alpha=(0.0)_beta=(0.0)\.pkl",
                                        lambda a,b: (float(a), float(b))),
    "FEN": lambda: _load_folder(FEN_DIR,  r"fairness_results\.pkl",            str),
    "SOTO": lambda: _load_folder(SOTO_DIR, r"fairness_results_alpha=(.+)\.pkl", str),
}

def generate_radar_plot(selected_algorithms, plot_title, output_filename):
    """
    Loads data for a given set of algorithms and generates a radar plot
    """
    
    data = {}
    all_actions = set()

    for algo, knob in selected_algorithms.items():
        store = LOADERS[algo]()
        if knob not in store:
            print(f"[warn] {algo} knob {knob} not found – skipped")
            continue

        freqs = store[knob]["action_frequencies"]
        if not {"non_sensitive", "sensitive"} <= freqs.keys():
            print(f"[warn] {algo} missing group keys – skipped")
            continue

        def normalise(d):
            tot = sum(d.values())
            return {k: (v / tot if tot else 0) for k, v in d.items()}

        ns, se = normalise(freqs["non_sensitive"]), normalise(freqs["sensitive"])
        data[algo] = {"ns": ns, "se": se}
        all_actions.update(ns.keys())

    if not data:
        print(f"No data loaded for '{plot_title}' – skipping plot.")
        return

    actions = sorted(all_actions)
    n_actions = len(actions)
    angles = np.linspace(0, 2*np.pi, n_actions, endpoint=False).tolist() + [0]

    # Plot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(actions, fontsize=11)

    max_val = max(max(max(g["ns"].values()), max(g["se"].values())) for g in data.values()) or 1.0
    ax.set_ylim(0, max_val*1.1)
    rticks = np.linspace(0, max_val, 5)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{v:.0%}" for v in rticks], size=9, color="grey")
    ax.grid(color="lightgrey", linestyle="dotted", linewidth=.8)

    for algo, g in data.items():
        col = ALGO_COLOURS[algo]
        ns_vals = [g["ns"].get(a, 0) for a in actions] + [g["ns"].get(actions[0], 0)]
        se_vals = [g["se"].get(a, 0) for a in actions] + [g["se"].get(actions[0], 0)]

        ax.plot(angles, ns_vals, color=col, linewidth=1.5, marker="o", label=f"{algo} (non-sens)")
        ax.fill(angles, ns_vals, color=col, alpha=.13)
        ax.plot(angles, se_vals, color=col, linewidth=1.0, linestyle="--", marker="^", label=f"{algo} (sens)")

    ax.set_title(plot_title, y=1.10, fontsize=16)

    alg_handles = [Line2D([0],[0], color=ALGO_COLOURS[a], lw=4) for a in selected_algorithms if a in data]
    group_handles = [
        Line2D([0],[0], color="k", lw=1.5, marker="o", label="non-sens", mfc="k", mec="k", ls='-'),
        Line2D([0],[0], color="k", lw=1.0, marker="^", label="sens", mfc="k", mec="k", ls='--')
    ]

    leg1 = ax.legend(alg_handles, [a for a in selected_algorithms if a in data],
                     title="Algorithm", loc="upper left",
                     bbox_to_anchor=(-0.05, 1.13), frameon=False, fontsize=10)
    leg2 = ax.legend(group_handles, ["non-sensitive", "sensitive"],
                     title="Group", loc="upper right",
                     bbox_to_anchor=(1.18, 1.13), frameon=False, fontsize=10)
    ax.add_artist(leg1)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, output_filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# main
if __name__ == "__main__":
    # Configuration for Fair-PPO Variants
    fair_ppo_selected = {
        "Fair-PPO (0.25, 0.0)": (0.25, 0.0),
        "Fair-PPO (0.25, 0.25)": (0.25, 0.25),
        "Fair-PPO (0.75, 0.25)": (0.75, 0.25),
    }
    
    # Configuration for other algorithms
    other_algs_selected = {
        "PPO": (0.0, 0.0),
        "FEN": "",
        "SOTO": "1.0",
    }

    generate_radar_plot(
        selected_algorithms=fair_ppo_selected,
        plot_title="Action Profiles of Fair-PPO Variants",
        output_filename="fair_ppo_variants_radar.svg"
    )

    generate_radar_plot(
        selected_algorithms=other_algs_selected,
        plot_title="Action Profiles of Baseline Algorithms",
        output_filename="baseline_algorithms_radar.svg"
    )
