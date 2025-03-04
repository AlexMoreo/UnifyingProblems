from glob import glob
import pandas as pd
import util
import matplotlib.pyplot as plt
import seaborn as sns

input_dir = './results/calibration/label_shift/repeats_100'

results = pd.concat([pd.read_csv(result_file) for result_file in glob(input_dir+'/*.csv')])

filter_out = [f'HDcal{n}-sm-mono' for n in [20,25,30,35]]+['PACC-cal']
for method in filter_out:
    results = results[results['method'] != method]

# pivot = results.pivot_table(index='dataset', columns='method', values='ece')

# results = results.reset_index(drop=True)
#
# sns.set_style("whitegrid")
#
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=results, x="shift", y="ece", hue="method", alpha=0.7)
#
# plt.yscale("log")
# plt.xlabel("Shift")
# plt.ylabel("log ECE")
# plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
# plt.tight_layout()
#
# plt.show()


num_bins = 20

results["shift_bin"] = pd.cut(results["shift"], bins=num_bins, labels=False)

summary = results.groupby(["shift_bin", "method"]).agg(
    mean_ece=("ece", "mean"),
    std_ece=("ece", "std"),
    shift_center=("shift", "mean")  # the bin center
).reset_index()

plt.figure(figsize=(8, 6))

sns.lineplot(
    data=summary,
    x="shift_center", y="mean_ece",
    hue="method",
    errorbar=("sd"),
    marker="o"
)

logscale = True

if logscale:
    plt.yscale("log")
    plt.ylabel("Mean ECE (log scale)")
else:
    plt.ylabel("Mean ECE")
plt.xlabel("Shift")
plt.title("Mean ECE vs Shift (binned)")

plt.legend(title="Method", loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.show()