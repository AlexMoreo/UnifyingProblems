import itertools
import os
from glob import glob
import pandas as pd
import util
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

from gen_tables import replace_method


def included_method(file_path, include_methods):
    name = Path(file_path).name.split('_')[0]
    for method in include_methods:
        if method == name:
            return True
    return False

tasks = ['classifier_accuracy_prediction', 'calibration', 'quantification']
dataset_shifts = ['covariate_shift', 'label_shift']

for task, dataset_shift in itertools.product(tasks, dataset_shifts):
    input_dir = f'./results/{task}/{dataset_shift}/repeats_100_samplesize_250'

    logscale = False
    if task == 'calibration':
        err = 'ece'
        err_name = err.upper()
        if dataset_shift == 'covariate_shift':
            include_methods = ['Platt', 'Head2Tail', 'CPCS', 'TransCal', 'LasCal', 'Bin6-PCC5', 'Bin6-PACC5', 'Bin2-DoC6']
            logscale=False
        else:
            include_methods = ['Platt', 'Head2Tail-P', 'CPCS-P', 'TransCal-S', 'LasCal-P', 'EM-BCTS', 'HDcal8-sm-mono']
    elif task == 'quantification':
        err = 'ae'
        err_name = err.upper()
        if dataset_shift == 'covariate_shift':
            include_methods = ['CC', 'PCC', 'PACC', 'EMQ', 'KDEy', 'DoC-q', 'LEAP-PCC-q', 'TransCal-q-P']
            # logscale = True
        else:
            include_methods = ['CC', 'PACC', 'EMQ', 'KDEy', 'DoC-q', 'LasCal-q-P']
            # logscale = True
    elif task == 'classifier_accuracy_prediction':
        err = 'err'
        err_name = 'AE'
        if dataset_shift == 'covariate_shift':
            include_methods = ['Naive', 'ATC', 'DoC', 'LEAP', 'LEAP-PCC', 'PCC-a', 'LasCal-a-P']
            # logscale = True
        else:
            include_methods = ['Naive', 'ATC', 'DoC', 'LEAP', 'Cpcs-a-S', 'PACC-a']
            # logscale = True
    custom_palette = {
        "Naive": "black",
        "Platt": "black",
        "Head2Tail": "blueviolet",
        "Head2Tail-P": "blueviolet",
        "CPCS": "firebrick",
        'CPCS-P': "firebrick",
        'Cpcs-a-S': "firebrick",
        'TransCal': "lightgreen",
        'TransCal-S': "lightgreen",
        'TransCal-q-P': "lightgreen",
        'LasCal': "gold",
        'LasCal-P': "gold",
        'LasCal-a-P': "gold",
        'LasCal-q-P': "gold",
        'CC': "black",
        'PCC': "gray",
        'PCC-a': "gray",
        'Bin6-PCC5': "gray",
        'PACC': "magenta",
        'PACC-a': "magenta",
        'Bin6-PACC5': "magenta",
        'EM': "dodgerblue",
        'EMQ': "dodgerblue",
        'EM-BCTS': "dodgerblue",
        "KDEy": "green",
        "HDcal8-sm-mono": "orangered",
        'ATC': "darkorange",
        'DoC': "blue",
        'DoC-q': "blue",
        'Bin2-DoC6': "blue",
        'LEAP': "cadetblue",
        'LEAP-PCC': "red",
        'LEAP-PCC-q': "red",
    }
    custom_palette = {replace_method.get(m,m): color for m, color in custom_palette.items()}


    results = pd.concat([pd.read_csv(result_file) for result_file in glob(input_dir+'/*.csv') if included_method(result_file, include_methods)])

    results['method'] = results['method'].replace(replace_method)
    methods_renamed = [replace_method.get(m,m) for m in include_methods]

    num_bins = 10
    bin_edges = np.linspace(0,1,num_bins+1)
    bin_edges[0]-=1e-4
    bin_edges[-1] += 1e-4
    results["shift_bin"] = pd.cut(results["shift"], bins=bin_edges, labels=False)

    summary = results.groupby(["shift_bin", "method"]).agg(
        mean_err=(err, "mean"),
        std_err=(err, "std"),
        shift_center=("shift", "mean")  # the bin center
    ).reset_index()

    if dataset_shift=='covariate_shift':
        bin_density = np.full(shape=num_bins, fill_value=1./num_bins)
    else:
        bin_counts = results["shift_bin"].value_counts().sort_index()
        bin_density = bin_counts / bin_counts.sum()  # density

    bin_centers = bin_edges[1:]-0.5/num_bins  # bin center

    max_density = np.max(bin_density)
    max_err = np.max(summary.mean_err)
    scale = max_err / max_density
    bin_density = bin_density * scale

    plt.figure(figsize=(10, 8))

    binwidth = 1/num_bins
    plt.bar(bin_centers, bin_density, width=binwidth-binwidth*0.1, color="lavender", alpha=0.9, label="Density")

    sns.lineplot(
        data=summary,
        x="shift_center", y="mean_err",
        hue="method",
        hue_order=methods_renamed,
        errorbar=("sd"),
        marker="o",
        linewidth=3,
        markersize=10,
        palette=custom_palette
    )

    if logscale:
        plt.yscale("log")
        plt.ylabel(f"{err_name} (log scale)")
    else:
        plt.ylabel(f"{err_name}")
    plt.xlabel("Level of shift")
    plt.title(f"{task.title().replace('_',' ')} under {dataset_shift.replace('_', ' ').title()}")
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

