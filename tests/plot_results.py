from matplotlib import pyplot as plt
import pandas as pd

from constants import *

TEST_DIR_NAME = "marine_ecology_2"

data = {
    "Model": [
        "gpt-4-turbo-2024-04-09_dspy",
        "gpt-4o-2024-05-13_dspy",
        "gpt-4-0613_dspy",
        "gpt-3.5-turbo-0125_dspy",
        "mistral7b:instruct",
        "gpt-4-turbo-2024-04-09_baseline",
        "gpt-4o-2024-05-13_baseline",
        "gpt-4-0613_baseline",
        "gpt-3.5-turbo-0125_baseline",
    ],
    "Precision": [
        0.5789473684210527,
        0.0,
        0.68,
        0.4032258064516129,
        0.8125,
        0.8695652173913043,
        0.8125,
        0.8163265306122449,
        0.7166666666666667,
    ],
    "Recall": [
        0.2,
        0.0,
        0.6181818181818182,
        0.45454545454545453,
        0.23636363636363636,
        0.7272727272727273,
        0.7090909090909091,
        0.7272727272727273,
        0.7818181818181819,
    ],
    "F1": [
        0.2972972972972973,
        0.0,
        0.6476190476190475,
        0.42735042735042733,
        0.36619718309859156,
        0.792079207920792,
        0.7572815533980584,
        0.7692307692307693,
        0.7478260869565218,
    ],
}

df = pd.DataFrame(data)

df_truncated = df.copy()
df_truncated[["Precision", "Recall", "F1"]] = df_truncated[
    ["Precision", "Recall", "F1"]
].round(3)

plt.figure(figsize=(5, 3))
plt.axis("off")
plt.title("Relationship Extraction Performance Metrics")
table = plt.table(
    cellText=df_truncated.values,
    colLabels=df_truncated.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.auto_set_column_width(col=list(range(len(df_truncated.columns))))
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.savefig(
    f"{SYNTHETIC_DIR}/{TEST_DIR_NAME}/performance_table_all.png",
    dpi=300,
    bbox_inches="tight",
)

ax = df_truncated.plot(kind="bar", x="Model", figsize=(14, 10))
plt.xticks(rotation=45, ha="right")
plt.ylabel("Scores")
plt.title("Relationship Extraction Performance Metrics")
plt.legend(loc="upper right")

# annotate bars with values
for container in ax.containers:
    ax.bar_label(container, label_type="edge", fmt="%.3f")

plt.tight_layout()
plt.savefig(f"{SYNTHETIC_DIR}/{TEST_DIR_NAME}/performance_graph_all.png", dpi=300)
