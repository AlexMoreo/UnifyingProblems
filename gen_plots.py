from glob import glob
import pandas as pd
import util

input_dir = './results/calibration/label_shift/repeats_100'

results = pd.concat([pd.read_csv(result_file) for result_file in glob(input_dir+'/*.csv')])

pivot = results.pivot_table(index='dataset', columns='method', values='ece')


import matplotlib.pyplot as plt
import seaborn as sns

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


# Definir número de bins (ajusta según necesites)
num_bins = 10

# Crear los bins para "shift"
results["shift_bin"] = pd.cut(results["shift"], bins=num_bins, labels=False)

# Calcular media y desviación estándar de ECE en cada bin, para cada método
summary = results.groupby(["shift_bin", "method"]).agg(
    mean_ece=("ece", "mean"),
    std_ece=("ece", "std"),
    shift_center=("shift", "mean")  # Usamos el centro del bin para graficar
).reset_index()

plt.figure(figsize=(8, 6))

# Graficar líneas con barras de error (desviación estándar)
sns.lineplot(
    data=summary,
    x="shift_center", y="mean_ece",
    hue="method",
    errorbar=("sd"),  # Muestra barras de error con la desviación estándar
    marker="o"
)

# plt.yscale("log")
plt.xlabel("Shift")
# plt.ylabel("Mean ECE (log scale)")
plt.ylabel("Mean ECE")
plt.title("Mean ECE vs Shift (binned)")

plt.legend(title="Method", loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()

plt.show()