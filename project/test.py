# Environmental Factors:
#
# 1) How many students live in conditions with high noise levels?
# 2) What percentage of students feel unsafe in their living conditions?
# 3) How many students have reported not having their basic needs met?
from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./StressLevelDataset.csv")


# 1) How many students live in conditions with high noise levels?
noise_level_count = OrderedDict()
for level in df["noise_level"]:
    noise_level_count[level] = noise_level_count.get(level, 0) + 1

plt.pie(list(noise_level_count.values()), labels=list(noise_level_count.keys()), autopct="%0.1f%%")
plt.title("Noise Level Percentage")
plt.show()


# 2) What percentage of students feel unsafe in their living conditions?
safety_level_count = OrderedDict()
for level in df["safety"]:
    safety_level_count[level] = safety_level_count.get(level, 0) + 1

plt.pie(list(safety_level_count.values()), labels=list(safety_level_count.keys()), autopct="%0.1f%%")
plt.title("Safety Level Percentage")
plt.show()


# 3) How many students have reported not having their basic needs met?
basic_needs_level_count = OrderedDict()
for level in df["basic_needs"]:
    basic_needs_level_count[level] = basic_needs_level_count.get(level, 0) + 1

plt.pie(list(basic_needs_level_count.values()), labels=list(basic_needs_level_count.keys()), autopct="%0.1f%%")
plt.title("Basic Needs Level Percentage")
plt.show()
