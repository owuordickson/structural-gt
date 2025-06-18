import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import cbook

np.random.seed(19680801)
data = np.random.randn(20, 3)
# print(data)

fig, (ax1, ax2) = plt.subplots(1, 2)

# single boxplot call
ax1.boxplot(data, tick_labels=['A', 'B', 'C'], patch_artist=True, boxprops={'facecolor': 'bisque'})


# plt.show()

# separate calculation of statistics and plotting
# stats = cbook.boxplot_stats(data, labels=['A', 'B', 'C'])
# ax2.bxp(stats, patch_artist=True, boxprops={'facecolor': 'bisque'})
