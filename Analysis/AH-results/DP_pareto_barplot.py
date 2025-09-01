#!/usr/bin/env python3

import os, re, pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path fairness_results (Fair-PPO, FEN, SOTO)
FAIRPPO_DIR = "ADD_PATH"
FEN_DIR = "ADD_PATH"
SOTO_DIR = "ADD_PATH"

output_dir = "ADD_PATH"
os.makedirs(output_dir, exist_ok=True)

ALGORITHMS = ["fair-PPO", "vanilla-PPO", "FEN", "SOTO"]
base_colors = {
    "fair-PPO": "#7570b3",
    "vanilla-PPO": "#66b3ff",
    "FEN": "#1b9e77",
    "SOTO": "#d95f02",
}

# alias 
ALIAS = {
    "able_red": "non_sensitive_red",
    "able_blue": "non_sensitive_blue",
    "disabled_red": "sensitive_red",
    "disabled_blue":"sensitive_blue",
}

def _canon(k): return ALIAS.get(k, k)

def extract_episode_lists(cumul):
    out = {g: list(cumul.get(g, [])) for g in (
        "non_sensitive_red", "non_sensitive_blue",
        "sensitive_red",     "sensitive_blue")}
    if "non_sensitive" in cumul:
        L = len(cumul["non_sensitive"])
        out["non_sensitive_red"], out["non_sensitive_blue"] = list(cumul["non_sensitive"]), [0]*L
    if "sensitive" in cumul:
        L = len(cumul["sensitive"])
        out["sensitive_red"], out["sensitive_blue"] = list(cumul["sensitive"]), [0]*L
    for k, v in cumul.items():
        if k in ALIAS:
            out[_canon(k)] = list(v)
    n_eps = max(len(v) for v in out.values()) if out else 0
    for g in out:
        out[g].extend([0]*(n_eps - len(out[g])))
    return out

# fairness metrics (equal reward distributions)
def gini(x): x = np.sort(np.asarray(x,float)); n=len(x); return 0 if x.sum()==0 else (2*(np.arange(1,n+1)*x).sum())/(n*x.sum())-(n+1)/n
def jain(x): x = np.asarray(x,float); s=x.sum(); return 0 if s==0 else s**2/(len(x)*(x**2).sum())
def nash_norm(x): x = np.asarray(x,float)+1e-10; return (x.prod()**(1/len(x)))/x.mean()

def per_episode_metrics(epl):
    dp,gi,jf,nn=[],[],[],[]
    for r_ns, b_ns, r_s, b_s in zip(
            epl["non_sensitive_red"], epl["non_sensitive_blue"],
            epl["sensitive_red"],     epl["sensitive_blue"]):
        rewards = np.array([r_ns,b_ns,r_s,b_s])
        dp.append((r_ns+b_ns)-(r_s+b_s))
        gi.append(gini(rewards)); jf.append(jain(rewards)); nn.append(nash_norm(rewards))
    return dp,gi,jf,nn

# load pickle results
def load_folder(folder, regex, cast):
    rx=re.compile(regex); out={}
    for f in os.listdir(folder):
        m=rx.match(f)
        if m:
            with open(os.path.join(folder,f),"rb") as fp:
                out[cast(*m.groups())]=pickle.load(fp)
    return out

# check file name formats for reproduction
ALL_FP = load_folder(
    FAIRPPO_DIR,
    r"fairness_results_alpha=([0-9.]+)_beta=([0-9.]+)\.pkl",
    lambda a,b:(float(a),float(b))
)
LOADERS = {
    "vanilla-PPO": lambda:{k:v for k,v in ALL_FP.items() if abs(k[0])<1e-9 and abs(k[1])<1e-9},
    "fair-PPO": lambda:{k:v for k,v in ALL_FP.items() if not(abs(k[0])<1e-9 and abs(k[1])<1e-9)},
    "FEN": lambda:load_folder(FEN_DIR , r"fairness_results\.pkl", str),
    "SOTO": lambda:load_folder(SOTO_DIR, r"fairness_results_alpha=(.+)\.pkl", str),
}

# build dataframe
rows=[]
for algo in ALGORITHMS:
    for knob,blob in LOADERS[algo]().items():
        epl=extract_episode_lists(blob["cumulative_rewards"])
        dp,gi,jf,nn=per_episode_metrics(epl)
        knob_str="" if algo in ("vanilla-PPO","FEN") else (
            f"α-fair={knob}" if algo=="SOTO" else f"α={knob[0]},β={knob[1]}")
        for a,b,c,d in zip(dp,gi,jf,nn):
            rows.extend([
                dict(Algorithm=algo,Knob=knob_str,Metric="DP_absolute",Value=a),
                dict(Algorithm=algo,Knob=knob_str,Metric="Gini", Value=b),
                dict(Algorithm=algo,Knob=knob_str,Metric="JFI", Value=c),
                dict(Algorithm=algo,Knob=knob_str,Metric="NNSW", Value=d),
            ])
long_df=pd.DataFrame(rows)

# normalise DP
max_dp=long_df.loc[long_df.Metric=="DP_absolute","Value"].abs().max()
if max_dp>0:
    long_df.loc[long_df.Metric=="DP_absolute","Value"]/=max_dp
long_df.loc[long_df.Metric=="DP_absolute","Metric"]="DP_scaled"

# pick knob with min DP
best={}
for algo in ALGORITHMS:
    sub=long_df[(long_df.Algorithm==algo)&(long_df.Metric=="DP_scaled")]
    if not sub.empty:
        best[algo]=sub.groupby("Knob").Value.mean().idxmin()
best_df=long_df[long_df.apply(lambda r:r.Knob==best.get(r.Algorithm,""),axis=1)]

# boxplots
metric_names={"DP_scaled":"Demographic Disparity","Gini":"Gini Index","NNSW":"Normalised Nash SW"}
plt.rcParams["font.family"]="serif"
fig,axes=plt.subplots(1,3,figsize=(12,4))
fig.suptitle("Best-DP configuration per algorithm",fontsize=16,fontweight="bold")
algo_palette=[base_colors[a] for a in ALGORITHMS]
for ax,(mkey,title) in zip(axes,metric_names.items()):
    sns.boxplot(ax=ax,data=best_df[best_df.Metric==mkey],
                x="Algorithm",y="Value",order=ALGORITHMS,
                palette=algo_palette,width=0.8)
    ax.set_title(title,fontsize=12,fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel(title,fontsize=11)
    ax.tick_params(axis="x",labelsize=10,rotation=15)
plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig(os.path.join(output_dir,"metrics_boxplots_best.svg")); plt.close()

# generate LaTex table
summary=(best_df.groupby(["Algorithm","Metric"]).Value.mean()
         .unstack("Metric").reset_index().rename(columns={
             "DP_scaled":"DP (↓)","Gini":"Gini (↓)","JFI":"JFI (↑)","NNSW":"NNSW (↑)"}))
summary=summary[["Algorithm","DP (↓)","Gini (↓)","JFI (↑)","NNSW (↑)"]]
summary["_o"]=summary.Algorithm.apply(ALGORITHMS.index)
summary=summary.sort_values("_o").drop(columns="_o")
with open(os.path.join(output_dir,"best_configs_metrics_table.tex"),"w") as fp:
    fp.write(summary.to_latex(index=False,float_format="%.3f",
                               caption="Mean metrics (best |DP|) per algorithm",
                               label="tab:best_configs_metrics",
                               column_format="lcccc",escape=False))

# Pareto frontier
def mean_total_reward(epl):
    r = (np.asarray(epl["non_sensitive_red"])  + np.asarray(epl["non_sensitive_blue"])
       + np.asarray(epl["sensitive_red"]) + np.asarray(epl["sensitive_blue"]))
    return r.mean()

pareto_rows = []
for algo in ALGORITHMS:
    for knob, blob in LOADERS[algo]().items():
        epl = extract_episode_lists(blob["cumulative_rewards"])
        dp, *_ = per_episode_metrics(epl)
        pareto_rows.append(dict(
            Algorithm = algo,
            Knob = ("" if algo in ("vanilla-PPO", "FEN") else (f"α-fair={knob}" if algo == "SOTO" else f"α={knob[0]},β={knob[1]}")),
            Fairness = np.abs(dp).mean(),
            Reward = mean_total_reward(epl)))

pareto_df = pd.DataFrame(pareto_rows)

fair_max = pareto_df["Fairness"].max()
rew_max = pareto_df["Reward"].max()

# protect against division-by-zero in degenerate cases
if fair_max > 0: pareto_df["Fairness_norm"] = pareto_df["Fairness"] / fair_max
else: 
    pareto_df["Fairness_norm"] = 0
if rew_max  > 0: pareto_df["Reward_norm"] = pareto_df["Reward"] / rew_max
else:            
    pareto_df["Reward_norm"] = 0

# build the frontier
frontier = (pareto_df.sort_values(["Fairness_norm", "Reward_norm"])
            .groupby("Fairness_norm", as_index=False)
            .max()
            .sort_values("Fairness_norm"))

plt.figure(figsize=(5.5, 3))
sns.scatterplot(data=pareto_df,
                x="Fairness_norm", y="Reward_norm",
                hue="Algorithm",  style="Algorithm",
                palette=base_colors, s=60, alpha=0.85)
plt.plot(frontier["Fairness_norm"], frontier["Reward_norm"],
         ls="--", lw=1, color="black", label="Pareto frontier")

plt.xlabel("Demographic Parity", fontsize=10)
plt.ylabel("Mean Reward", fontsize=10)
plt.title("Pareto frontier: Mean Reward vs Fairness",
          fontsize=16, fontweight="bold")
plt.legend(title="", fontsize=10)
plt.tight_layout()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "pareto_frontier_reward_vs_fairness.svg"))
plt.close()

# Group-reward bar chart (best-DP)
def knobstr_to_key(algo,knob_str):
    """
    Map pretty knob label
    """
    
    if algo=="FEN":
        return ""
    if algo=="vanilla-PPO":
        return (0.0,0.0)
    if algo=="SOTO":         
        return knob_str.replace("α-fair=","")
    # fair-PPO
    m=re.match(r"α=([0-9.]+),β=([0-9.]+)",knob_str)
    if not m: 
        raise KeyError(f"Cannot parse knob label '{knob_str}'")
    return (float(m[1]),float(m[2]))

bar_rows=[]
for algo in ALGORITHMS:
    knob_str=best.get(algo,"")
    loader = LOADERS[algo]()
    try:
        key=knobstr_to_key(algo,knob_str)
        blob=loader[key]
    except Exception:
        blob=next(iter(loader.values()))
    epl=extract_episode_lists(blob["cumulative_rewards"])
    non_s=np.asarray(epl["non_sensitive_red"])+np.asarray(epl["non_sensitive_blue"])
    sens =np.asarray(epl["sensitive_red"])+np.asarray(epl["sensitive_blue"])
    bar_rows.extend([
        dict(Algorithm=algo,Group="Non-sensitive",Reward=non_s.mean()),
        dict(Algorithm=algo,Group="Sensitive",    Reward=sens.mean()),
    ])
bar_df=pd.DataFrame(bar_rows)
plt.figure(figsize=(7,4.5))
sns.barplot(data=bar_df,x="Algorithm",y="Reward",hue="Group",order=ALGORITHMS,palette=["#4c72b0","#dd8452"],width=0.8)
plt.xlabel(""); plt.ylabel("Mean episode reward")
plt.title("Average rewards by group (best-DP config.)",fontsize=14,fontweight="bold")
plt.legend(title=""); plt.xticks(rotation=15); plt.tight_layout()
plt.savefig(os.path.join(output_dir,"group_reward_barchart.pdf"))
plt.close()