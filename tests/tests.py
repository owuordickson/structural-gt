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


def check_for_updates():
    import requests
    from packaging import version

    __version__ = "2.3.7"
    github_url = "https://raw.githubusercontent.com/compass-stc/StructuralGT/refs/heads/DicksonOwuor-GUI/src/StructuralGT/__init__.py"

    try:
        response = requests.get(github_url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        msg = f"Error checking for updates: {e}"
        return msg

    remote_version = None
    for line in response.text.splitlines():
        if line.strip().startswith("__version__"):
            try:
                remote_version = line.split("=")[1].strip().strip("\"'")
                break
            except IndexError:
                msg = "Could not connect to server!"
                return msg

    if not remote_version:
        msg = "Could not find the new version!"
        return msg

    new_version = version.parse(remote_version)
    current_version = version.parse(__version__)
    if new_version > current_version:
        msg = (
            "New version available!<br>"
            "Download via this <a href='https://forms.gle/oG9Gk2qbmxooK63D7'>link</a>"
        )
    else:
        msg = "No updates available."
    return msg


print(check_for_updates())
