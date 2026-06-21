import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline


sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,

    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],

    #   TITLE (BIG)
    "axes.titlesize": 18,
    "axes.titleweight": "bold",

    #   AXIS LABELS (BIG)
    "axes.labelsize": 16,
    "axes.labelweight": "bold",

    #   TICK LABELS (X & Y)
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    #   LEGEND TEXT
    "legend.fontsize": 14
})

palette = sns.color_palette("deep")


def smooth_curve(x, y, points=300, k=2):

    x = np.array(x)
    y = np.array(y)

    # Create smooth x-axis
    x_smooth = np.linspace(x.min(), x.max(), points)

    # Spline interpolation
    spline = make_interp_spline(x, y, k=k)

    # Smooth y-axis
    y_smooth = spline(x_smooth)

    return x_smooth, y_smooth



with open("RESULTS_AVG.json", "r") as f:
    thesis = sorted(json.load(f), key=lambda x: x["num_tasks"])

with open("RESULTS_Base_Paper_Extended_AVG.json", "r") as f:
    base = sorted(json.load(f), key=lambda x: x["num_tasks"])

tasks = [d["num_tasks"] for d in thesis]

rt_t = [d["avg_response_time"] for d in thesis]
rt_b = [d["avg_response_time"] for d in base]

dmr_t = [d["deadline_miss_ratio"] for d in thesis]
dmr_b = [d["deadline_miss_ratio"] for d in base]

energy_t = [d["total_energy"] for d in thesis]
energy_b = [d["total_energy"] for d in base]

ms_t = [d["mode_switches"] for d in thesis]
ms_b = [d["mode_switches"] for d in base]

blocking_t = [d["total_blocking_time"] for d in thesis]
blocking_b = [d["total_blocking_time"] for d in base]

# =========================================================
#   RESPONSE TIME
# =========================================================
plt.figure(figsize=(9, 6))

x_smooth, y_smooth = smooth_curve(tasks, rt_b)

# plt.plot(x_smooth, y_smooth,
#          linewidth=3,
#          color=palette[0],
#          label="PMCS-HLP")

# Original points
plt.scatter(tasks, rt_b,
            s=70,
            color=palette[0])

plt.plot(tasks, rt_b,
         marker="o", linewidth=3,
         color=palette[0],
         label="PMCS-HLP")

x_smooth, y_smooth = smooth_curve(tasks, rt_t)

plt.plot(tasks, rt_t,
         marker="o", linewidth=3,
         color=palette[1],
         label="PMC-HM")

# plt.plot(x_smooth, y_smooth,
#          linewidth=3,
#          color=palette[1],
#          label="PMC-HM")

plt.scatter(tasks, rt_t,
            s=70,
            color=palette[1])

plt.title("Average Response Time Comparison", pad=12)
plt.xlabel("Number of Tasks")
plt.ylabel("Response Time (seconds)")
plt.xticks(tasks)
plt.legend()

#plt.gca().set_box_aspect(1)
plt.tight_layout()


# =========================================================
#   DEADLINE MISS RATIO
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, dmr_b,
         marker="o", linewidth=3,
         color=palette[0],
         label="PMCS-HLP")

plt.plot(tasks, dmr_t,
         marker="o", linewidth=3,
         color=palette[1],
         label="PMC-HM")

plt.title("Deadline Miss Ratio Comparison", pad=12)
plt.xlabel("Number of Tasks")
plt.ylabel("Miss Ratio")
plt.xticks(tasks)
plt.legend()

plt.gca().set_box_aspect(1)
plt.tight_layout()

# =========================================================
#   ENERGY
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, energy_b,
         marker="o", linewidth=3,
         color=palette[2],
         label="PMCS-HLP")

plt.plot(tasks, energy_t,
         marker="o", linewidth=3,
         color=palette[3],
         label="PMC-HM")

plt.title("Energy Consumption Comparison", pad=12)
plt.xlabel("Number of Tasks")
plt.ylabel("Total Energy (Joules)")
plt.xticks(tasks)
plt.legend()

#plt.gca().set_box_aspect(1)
plt.tight_layout()

# =========================================================
#   MODE SWITCHES
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, ms_b,
         marker="o", linewidth=3,
         color=palette[4],
         label="PMCS-HLP")

plt.plot(tasks, ms_t,
         marker="o", linewidth=3,
         color=palette[5],
         label="PMC-HM")

plt.title("Mode Switches Comparison", pad=12)
plt.xlabel("Number of Tasks")
plt.ylabel("Mode Switches Count")
plt.xticks(tasks)
plt.legend()
plt.tight_layout()

#plt.gca().set_box_aspect(1)

# =========================================================
#   Blocking Time
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, blocking_b,
         marker="o", linewidth=3,
         color=palette[4],
         label="PMCS-HLP")

plt.plot(tasks, blocking_t,
         marker="o", linewidth=3,
         color=palette[5],
         label="PMC-HM")

plt.title("Blocking Time Comparison", pad=12)
plt.xlabel("Number of Tasks")
plt.ylabel("Blocking Time (seconds)")
plt.xticks(tasks)
plt.legend()
plt.tight_layout()

#plt.gca().set_box_aspect(1)
#plt.show()



# =========================================================
#   LOAD DATA
# =========================================================
with open("RESULTS_RandomCores_AVG.json", "r") as f:
    thesis = sorted(json.load(f), key=lambda x: x["num_cores"])

with open("RESULTS_Base_Paper_RandomCores_AVG.json", "r") as f:
    base = sorted(json.load(f), key=lambda x: x["num_cores"])

tasks = [d["num_cores"] for d in thesis]

rt_t = [d["avg_response_time"] for d in thesis]
rt_b = [d["avg_response_time"] for d in base]

dmr_t = [d["deadline_miss_ratio"] for d in thesis]
dmr_b = [d["deadline_miss_ratio"] for d in base]

energy_t = [d["total_energy"] for d in thesis]
energy_b = [d["total_energy"] for d in base]

ms_t = [d["mode_switches"] for d in thesis]
ms_b = [d["mode_switches"] for d in base]

blocking_t = [d["total_blocking_time"] for d in thesis]
blocking_b = [d["total_blocking_time"] for d in base]

# =========================================================
#   RESPONSE TIME
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, rt_b,
         marker="o", linewidth=3,
         color=palette[0],
         label="PMCS-HLP")

plt.plot(tasks, rt_t,
         marker="o", linewidth=3,
         color=palette[1],
         label="PMC-HM")

plt.title("Average Response Time Comparison", pad=12)
plt.xlabel("Number of Cores")
plt.ylabel("Response Time (seconds)")
plt.xticks(tasks)
plt.legend()

#plt.gca().set_box_aspect(1)
plt.tight_layout()


# =========================================================
#   DEADLINE MISS RATIO
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, dmr_b,
         marker="o", linewidth=3,
         color=palette[0],
         label="PMCS-HLP")

plt.plot(tasks, dmr_t,
         marker="o", linewidth=3,
         color=palette[1],
         label="PMC-HM")

plt.title("Deadline Miss Ratio Comparison", pad=12)
plt.xlabel("Number of Cores")
plt.ylabel("Miss Ratio")
plt.xticks(tasks)
plt.legend()

#plt.gca().set_box_aspect(1)
plt.tight_layout()

# =========================================================
#   ENERGY
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, energy_b,
         marker="o", linewidth=3,
         color=palette[2],
         label="PMCS-HLP")

plt.plot(tasks, energy_t,
         marker="o", linewidth=3,
         color=palette[3],
         label="PMC-HM")

plt.title("Energy Consumption Comparison", pad=12)
plt.xlabel("Number of Cores")
plt.ylabel("Total Energy (Joules)")
plt.xticks(tasks)
plt.legend()

#plt.gca().set_box_aspect(1)
plt.tight_layout()

# =========================================================
#   MODE SWITCHES
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, ms_b,
         marker="o", linewidth=3,
         color=palette[4],
         label="PMCS-HLP")

plt.plot(tasks, ms_t,
         marker="o", linewidth=3,
         color=palette[5],
         label="PMC-HM")

plt.title("Mode Switches Comparison", pad=12)
plt.xlabel("Number of Cores")
plt.ylabel("Mode Switches (Count)")
plt.xticks(tasks)
plt.legend()
plt.tight_layout()

#plt.gca().set_box_aspect(1)

# =========================================================
#   Blocking Time
# =========================================================
plt.figure(figsize=(9, 6))

plt.plot(tasks, blocking_b,
         marker="o", linewidth=3,
         color=palette[4],
         label="PMCS-HLP")

plt.plot(tasks, blocking_t,
         marker="o", linewidth=3,
         color=palette[5],
         label="PMC-HM")

plt.title("Blocking Time Comparison", pad=12)
plt.xlabel("Number of Cores")
plt.ylabel("Blocking Time (seconds)")
plt.xticks(tasks)
plt.legend()
plt.tight_layout()

#plt.gca().set_box_aspect(1)
plt.show()
