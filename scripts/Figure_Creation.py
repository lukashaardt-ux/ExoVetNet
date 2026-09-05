import time

import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

labels = ['Catalog folding', 'BLS folding']
means = [0.9074, 0.8610]
stds  = [0.0049, 0.0125]
colors = ["#8B0000", "navy"]
plt.bar(labels, means, yerr=stds, capsize=6, color = colors)
plt.ylabel('F1 score')
plt.ylim(0.8, 0.95)

plt.savefig(f"figures/boxplot{time.time()}.png")

plt.show()

