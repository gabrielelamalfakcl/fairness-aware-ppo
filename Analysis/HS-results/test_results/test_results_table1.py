#!/usr/bin/env python3
# Fair-RL — Data Verification, Final Summary, and Plotting
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

def load_folder(folder, regex, cast):
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

def verify_and_process_blob(
    blob,
    run_name: str,
    summary_list: List[Dict],
):
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

    print("Starting data verification and processing")

    ppo_runs = load_folder(FAIRPPO_DIR, r"fairness_results_([0-9.]+)_([0-9.]+)\.pkl", lambda a, b: (float(a), float(b)))
    for (alpha, beta), blob in ppo_runs.items():
        run_name = "Vanilla-PPO" if alpha == 0.0 and beta == 0.0 else f"Fair-PPO_alpha={alpha}_beta={beta}"
        verify_and_process_blob(blob, run_name, summary_mean_rows)

    soto_runs = load_folder(SOTO_DIR, r"soto_results_.+?_alpha=([\d.]+)\.pkl", str)
    for alpha, blob in soto_runs.items():
        run_name = f"SOTO_alpha={alpha}"
        verify_and_process_blob(blob, run_name, summary_mean_rows)

    fen_runs = load_folder(FEN_DIR, r"fen_results\.pkl", lambda *a: "FEN_run")
    for _, blob in fen_runs.items():
        verify_and_process_blob(blob, "FEN", summary_mean_rows)

    if not summary_mean_rows:
        print("\nNo data was processed. Exiting.")
        exit()

    final_display_df = pd.DataFrame(summary_mean_rows).set_index('Run')

    TABLE_METRICS_CONFIG = {
        "total_patients_finished": {"header": r"{\makecell{\textbf{Patients} \\ \textbf{Treated} \\ \textbf{(Daily Average)}}}", "siunitx": "S[table-format=3.2]"},
        "reward_fairness_gap": {"header": r"{\makecell{\textbf{Dem} \\ \textbf{Disparity} \\ \textbf{(Rewards)}}}", "siunitx": "S[table-format=2.2]"},
        "escort_avg_patient_wait_time": {"header": r"{\makecell{\textbf{Patient} \\ \textbf{Wait Escort} \\ \textbf{(Minutes)}}}", "siunitx": "S[table-format=1.2]"},
        "escort_avg_helper_travel_time": {"header": r"{\makecell{\textbf{Escort} \\ \textbf{Travel Time} \\ \textbf{(Minutes)}}}", "siunitx": "S[table-format=1.2]"},
        "manager_doctor_moves": {"header": r"{\makecell{\textbf{Swing Doctor} \\ \textbf{Moves} \\ \textbf{(Average)}}}", "siunitx": "S[table-format=4.2]"},
        "routing_perfect_pct": {"header": r"{\makecell{\textbf{Perfect Routing} \\ (\textbf{\% of Patients)}}}", "siunitx": "S[table-format=2.2]"},
        "routing_acceptable_pct": {"header": r"{\makecell{\textbf{Backup Routing} \\ \textbf{(\% of Patients)}}}", "siunitx": "S[table-format=2.2]"},
        "routing_wrong_pct": {"header": r"{\makecell{\textbf{Incorrect Routing} \\ \textbf{(\% of Patients)}}}", "siunitx": "S[table-format=2.2]"},
    }

    all_run_names = final_display_df.index.tolist()
    fairppo_runs = sorted([r for r in all_run_names if r.startswith("Fair-PPO")])
    vanilla_runs = [r for r in all_run_names if r == "Vanilla-PPO"]
    fen_runs = [r for r in all_run_names if r == "FEN"]
    soto_runs = sorted([r for r in all_run_names if r.startswith("SOTO")])
    ordered_runs_for_table = fairppo_runs + vanilla_runs + fen_runs + soto_runs

    col_specifiers = "l " + " ".join([cfg['siunitx'] for cfg in TABLE_METRICS_CONFIG.values()])
    headers = ["\\textbf{Algorithm}"] + [cfg['header'] for cfg in TABLE_METRICS_CONFIG.values()]
    metric_keys_ordered = list(TABLE_METRICS_CONFIG.keys())

    latex_string = r"""\begin{table*}
\centering
\sisetup{table-format = 4.2}
\resizebox{\textwidth}{!}{%
"""
    latex_string += f"\\begin{{tabular}}{{ {col_specifiers} }}\n"
    latex_string += "\\toprule\n"
    latex_string += " & ".join(headers) + " \\\\\n"
    latex_string += "\\midrule\n"

    previous_family = None
    for run_name in ordered_runs_for_table:
        current_family = run_name.split('_')[0]
        if previous_family is not None and current_family != previous_family:
            latex_string += "\\midrule\n"

        display_name = format_run_name_for_latex(run_name)
        row_data = final_display_df.loc[run_name]
        row_values = [display_name]
        for key in metric_keys_ordered:
            value = row_data.get(key, np.nan)
            if pd.isna(value):
                row_values.append("{-}")
            else:
                if "_pct" in key:
                    if abs(value) <= 1.0:
                        value *= 100
                row_values.append(f"{value:.2f}")

        latex_string += " & ".join(row_values) + " \\\\\n"
        previous_family = current_family

    latex_string += r"""\bottomrule
\end{tabular}}%
"""
    latex_string += r"""\caption{HospitalSim. Average efficiency (patients treated) and fairness performance (demographic disparity) across Fair-PPO and benchmark models. Further metrics are reported to complete the analysis.}
\label{HS-test-results-full}
\end{table*}"""

    latex_output_path = os.path.join(OUTPUT_DIR, "final_results_table1.tex")
    with open(latex_output_path, "w") as f:
        f.write(latex_string)
