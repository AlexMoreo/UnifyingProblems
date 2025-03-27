import itertools
import os
from glob import glob
import pandas as pd
import util
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def included_method(file_path, include_methods):
    name = Path(file_path).name.split('_')[0]
    for method in include_methods:
        if method == name:
            return True
    return False

tasks = ['calibration', 'classifier_accuracy_prediction', 'quantification']
dataset_shifts = ['label_shift', 'covariate_shift']

for task, dataset_shift in itertools.product(tasks, dataset_shifts):
    input_dir = f'./results/{task}/{dataset_shift}/repeats_100_samplesize_250'

    logscale = False
    if task == 'calibration':
        err = 'ece'
        err_name = err.upper()
        if dataset_shift == 'covariate_shift':
            include_methods = ['Platt', 'CPCS', 'TransCal', 'LasCal', 'EM', 'HDcal8-sm-mono']
            logscale=True
        else:
            include_methods = ['Platt', 'CPCS-S', 'TransCal-S', 'LasCal-S', 'EM', 'HDcal8-sm-mono']
    elif task == 'quantification':
        err = 'ae'
        err_name = err.upper()
        if dataset_shift == 'covariate_shift':
            include_methods = ['CC', 'PCC', 'PACC', 'EMQ', 'EMQ-BCTS', 'KDEy', 'ATC-q', 'DoC-q', 'LEAP-q', 'LasCal-q-P', 'TransCal-q-P', 'Cpcs-q-P']
            logscale = True
        else:
            include_methods = ['CC', 'PCC', 'PACC', 'EMQ', 'KDEy', 'ATC-q', 'DoC-q', 'LEAP-q', 'LasCal-q-P', 'TransCal-q-P', 'Cpcs-q-P']
            logscale = True
    elif task == 'classifier_accuracy_prediction':
        err = 'err'
        err_name = 'AE'
        if dataset_shift == 'covariate_shift':
            include_methods = ['Naive', 'ATC', 'DoC', 'LEAP', 'LEAP-PCC', 'TransCal-a-S', 'Cpcs-a-S', 'LasCal-a-P', 'PCC-a', 'PACC-a', 'KDEy-a', 'EMQ-BCTS-a']
            logscale = True
        else:
            include_methods = ['Naive', 'ATC', 'DoC', 'LEAP', 'TransCal-a-S', 'Cpcs-a-S', 'LasCal-a-P', 'PACC-a', 'KDEy-a', 'EMQ-a']
            # logscale = True



    results = pd.concat([pd.read_csv(result_file) for result_file in glob(input_dir+'/*.csv') if included_method(result_file, include_methods)])

    num_bins = 10
    bin_edges = np.linspace(0,1,num_bins+1)
    results["shift_bin"] = pd.cut(results["shift"], bins=bin_edges, labels=False)

    summary = results.groupby(["shift_bin", "method"]).agg(
        mean_ece=(err, "mean"),
        std_ece=(err, "std"),
        shift_center=("shift", "mean")  # the bin center
    ).reset_index()

    bin_counts = results["shift_bin"].value_counts().sort_index()
    bin_centers = bin_edges[1:]-0.5/num_bins #results.groupby("shift_bin")["shift"].mean()  # bin center
    bin_density = bin_counts / bin_counts.sum()  # density

    max_density = np.max(bin_density)
    max_ece = np.max(summary.mean_ece)
    scale = max_ece / max_density
    bin_density = bin_density * scale

    plt.figure(figsize=(8, 6))

    plt.bar(bin_centers, bin_density, width=0.09, color="gray", alpha=0.3, label="Density")

    sns.lineplot(
        data=summary,
        x="shift_center", y="mean_ece",
        hue="method",
        errorbar=("sd"),
        marker="o"
    )

    if logscale:
        plt.yscale("log")
        plt.ylabel(f"{err_name} (log scale)")
    else:
        plt.ylabel(f"{err_name}")
    plt.xlabel("Shift")
    plt.title(f"{task.title()} under {dataset_shift.replace('_', ' ').title()}")
    plt.xlim(0,1)
    plt.grid(True, which="both", axis='y', linestyle="--", linewidth=0.5, alpha=0.7)
    plt.minorticks_on()

    plt.legend(title="Method", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()

    outpath = f'figures/{task}_{dataset_shift}_err_by_shift.pdf'
    parent = Path(outpath).parent
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(outpath)