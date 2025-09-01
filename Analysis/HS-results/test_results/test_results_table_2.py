#!/usr/bin/env python3
# Fair-RL — Data Verification, Final Summary, and Plotting for CSP
# Author:  (your name)
# Date:  2025-08-04

import os
import re
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

# Path fairness_results/output
BASE_DIR = "ADD_PATH"
FAIRPPO_DIR = os.path.join(BASE_DIR, "ADD_PATH")
FEN_DIR = os.path.join(BASE_DIR, "ADD_PATH")
SOTO_DIR = os.path.join(BASE_DIR, "ADD_PATH")

OUTPUT_DIR = os.path.join(BASE_DIR, "ADD_PATH")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_folder(folder: str, regex: str, cast) -> Dict:
    rx = re.compile(regex)
    out = {}
    if not os.path.isdir(folder): return {}
    for f in os.listdir(folder):
        m = rx.match(f)
        if m:
            try:
                with open(os.path.join(folder, f), "rb") as fp:
                    out[cast(*m.groups())] = pickle.load(fp)
            except (pickle.UnpicklingError, EOFError):
                print(f"Warning: Could not load {f}.")
    return out

def verify_and_process_blob(blob, run_name: str, summary_list: List[Dict]):
    df = None
    if isinstance(blob, list) and blob: df = pd.DataFrame(blob)
    elif isinstance(blob, pd.DataFrame): df = blob

    print("\n" + "─" * 25 + f" Verifying: {run_name} " + "─" * 25)
    if df is None or df.empty:
        print("No data found in this file.")
        return

    numeric_df = df.select_dtypes(include=np.number)
    print(numeric_df.describe().to_markdown(floatfmt=".3f"))

    mean_values = numeric_df.mean().to_dict()
    mean_values['Run'] = run_name
    summary_list.append(mean_values)

def format_run_name_for_latex(run_name: str) -> str:
    if run_name.startswith("Fair-PPO"):
        match = re.search(r"alpha=([\d.]+)_beta=([\d.]+)", run_name)
        if match:
            alpha, beta = match.groups()
            return fr"Fair-PPO $\alpha={alpha}, \beta={beta}$"
    elif run_name.startswith("SOTO"):
        match = re.search(r"alpha=([\d.]+)", run_name)
        if match:
            alpha = match.group(1)
            return fr"SOTO $\alpha\text{{-fair}}={alpha}$"
    elif run_name == "Vanilla-PPO":
        return "PPO"
    return run_name.replace('_', ' ')


if __name__ == "__main__":
    summary_mean_rows = []

    # Load Fair-PPO and Vanilla-PPO runs
    ppo_runs = load_folder(FAIRPPO_DIR, r"fairness_results_csp_alpha=([0-9.]+)_beta=([0-9.]+)\.pkl", lambda a, b: (float(a), float(b)))
    for (alpha, beta), blob in ppo_runs.items():
        run_name = "Vanilla-PPO" if alpha == 0.0 and beta == 0.0 else f"Fair-PPO_alpha={alpha}_beta={beta}"
        verify_and_process_blob(blob, run_name, summary_mean_rows)

    # Load SOTO runs
    soto_runs = load_folder(SOTO_DIR, r"soto_results_.+?_alpha=([\d.]+)\.pkl", str)
    for alpha, blob in soto_runs.items():
        run_name = f"SOTO_alpha={alpha}"
        verify_and_process_blob(blob, run_name, summary_mean_rows)

    # Load FEN runs
    fen_runs = load_folder(FEN_DIR, r"fen_results\.pkl", lambda *a: "FEN_run")
    for _, blob in fen_runs.items():
        verify_and_process_blob(blob, "FEN", summary_mean_rows)

    if not summary_mean_rows:
        print("\nNo data was processed. Exiting.")
        exit()

    final_display_df = pd.DataFrame(summary_mean_rows).set_index('Run')

    # 1. Define the exact order of metric columns for the table
    METRIC_KEYS_IN_ORDER = [
        "reward_fairness_gap_HP", "reward_fairness_gap_MP", "reward_fairness_gap_LP",
        "finished_impaired_HP_count", "finished_non_impaired_HP_count",
        "finished_impaired_MP_count", "finished_non_impaired_MP_count",
        "finished_impaired_LP_count", "finished_non_impaired_LP_count",
    ]

    # 2. Sort all processed runs by algorithm family
    all_run_names = final_display_df.index.tolist()
    fairppo_runs = sorted([r for r in all_run_names if r.startswith("Fair-PPO")])
    vanilla_runs = [r for r in all_run_names if r == "Vanilla-PPO"]
    fen_runs = [r for r in all_run_names if r == "FEN"]
    soto_runs = sorted([r for r in all_run_names if r.startswith("SOTO")])
    ordered_runs_for_table = fairppo_runs + vanilla_runs + fen_runs + soto_runs

    # 3. Define the LaTeX table structure
    col_specifiers = (
        "l "
        "S[table-format=1.2] S[table-format=1.2] S[table-format=1.2] "
        "S[table-format=2.2] S[table-format=2.2] "
        "S[table-format=2.2] S[table-format=2.2] "
        "S[table-format=2.2] S[table-format=2.2]"
    )
    
    # Manually define the complex multi-level headers
    header_row1 = r"& \multicolumn{3}{c}{\textbf{Reward Gap}} & \multicolumn{6}{c}{\textbf{Patients Finished}} \\"
    cmidrule1   = r"\cmidrule(lr){2-4} \cmidrule(lr){5-10}"
    header_row2 = r"\textbf{Algorithm} & {\textbf{High}} & {\textbf{Medium}} & {\textbf{Low}} & \multicolumn{2}{c}{\textbf{High Prio}} & \multicolumn{2}{c}{\textbf{Medium Prio}} & \multicolumn{2}{c}{\textbf{Low Prio}} \\"
    cmidrule2   = r"\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}"
    header_row3 = r"& & & & {\textbf{Sens.}} & {\textbf{Non-Sens.}} & {\textbf{Sens.}} & {\textbf{Non-Sens.}} & {\textbf{Sens.}} & {\textbf{Non-Sens.}} \\"

    # 4. Build the LaTeX string
    latex_string = r"""\begin{table*}
\centering
\resizebox{\textwidth}{!}{%
"""
    latex_string += f"\\begin{{tabular}}{{ {col_specifiers} }}\n"
    latex_string += "\\toprule\n"
    latex_string += f"{header_row1}\n{cmidrule1}\n{header_row2}\n{cmidrule2}\n{header_row3}\n"
    latex_string += "\\midrule\n"

    # 5. Populate the table with all runs in the sorted order
    previous_family = None
    for run_name in ordered_runs_for_table:
        if run_name not in final_display_df.index:
            print(f"Warning: Run '{run_name}' not found in data. Skipping.")
            continue
        
        current_family = run_name.split('_')[0]
        if previous_family is not None and current_family != previous_family:
            latex_string += "\\midrule\n"

        display_name = format_run_name_for_latex(run_name)
        row_data = final_display_df.loc[run_name]
        row_values = [display_name]
        for key in METRIC_KEYS_IN_ORDER:
            value = row_data.get(key, np.nan)
            row_values.append("{-}" if pd.isna(value) else f"{value:.2f}")

        latex_string += " & ".join(row_values) + " \\\\\n"
        previous_family = current_family

    latex_string += r"""\bottomrule
\end{tabular}}%
"""
    latex_string += r"""\caption{HospitalSim with CSP. Comparison of mean reward gaps and patients finished across different priority levels for Fair-PPO and benchmark models.}
\label{HS-test-results-csp}
\end{table*}"""

    # 6. Save the final string to a file
    latex_output_path = os.path.join(OUTPUT_DIR, "final_results_table2.tex")
    with open(latex_output_path, "w") as f:
        f.write(latex_string)
