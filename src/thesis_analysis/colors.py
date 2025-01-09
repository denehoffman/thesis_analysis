import matplotlib as mpl
from matplotlib.colors import CenteredNorm, LinearSegmentedColormap

red = '#e41a1c'
light_red = '#f28c8d'
pink = '#f781bf'
orange = '#ff7f00'
yellow = '#ffff33'
purple = '#984ea3'
green = '#4daf4a'
blue = '#377eb8'
light_blue = '#97bfe0'
brown = '#a65628'
gray = '#999999'
black = '#000000'
white = '#ffffff'

cmap = LinearSegmentedColormap.from_list(
    'CustomColormap',
    [
        (0.0, blue),
        (0.497, green),
        (0.498, white),
        (0.5, white),
        (0.502, white),
        (0.60, yellow),
        (0.65, orange),
        (0.70, red),
        (1.0, black),
    ],
)
mpl.colormaps.register(cmap)


def get_centered_norm() -> CenteredNorm:
    return CenteredNorm()
