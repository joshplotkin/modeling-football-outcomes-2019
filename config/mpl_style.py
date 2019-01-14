from cycler import cycler
from matplotlib import rcParams
import seaborn as sns

## MATPLOTLIB PARAMETERS
rcParams['text.color'] = '#262626'
rcParams['font.size'] = 14.0

rcParams['axes.axisbelow'] = True
rcParams['axes.labelcolor'] = '#262626'
rcParams['axes.labelsize'] = 14
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.linewidth'] = 3.0
rcParams['axes.grid'] = True
rcParams['axes.facecolor'] = '#f0f0f0'
rcParams['axes.edgecolor'] = '#f0f0f0'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.titlesize'] = 16

rcParams['figure.dpi'] = 96
rcParams['figure.facecolor'] =  '#f0f0f0'
rcParams['figure.figsize'] = (12,8)
rcParams['figure.titlesize'] = 14
rcParams['figure.titleweight'] = 'bold'
rcParams['figure.subplot.left'] =  0.08
rcParams['figure.subplot.right'] =  0.95
rcParams['figure.subplot.bottom'] =  0.07

rcParams['grid.linestyle'] =  '-'
rcParams['grid.linewidth'] =  1.0
rcParams['grid.color'] =  '#cbcbcb'

rcParams['legend.fontsize'] = 12
rcParams['legend.fancybox'] = True

rcParams['lines.color'] = '#262626'
rcParams['lines.linewidth'] = 4
rcParams['lines.solid_capstyle'] = 'butt'

rcParams['patch.edgecolor'] = '#f0f0f0'
rcParams['patch.linewidth'] = 0.5

rcParams['savefig.edgecolor'] =  '#f0f0f0'
rcParams['savefig.facecolor'] =  '#f0f0f0'

rcParams['svg.fonttype'] =  'path'

rcParams['xtick.major.size'] =  0
rcParams['xtick.minor.size'] =  0
rcParams['ytick.major.size'] =  0
rcParams['ytick.minor.size'] =  0

## colors
color_palette = 'Set2'
colors = sns.color_palette(color_palette)
sns.palplot(sns.color_palette(color_palette))
rcParams['axes.prop_cycle'] = cycler(color=colors)