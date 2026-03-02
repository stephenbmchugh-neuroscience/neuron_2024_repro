##############################################################
## Import libraries
##############################################################
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.ticker as Ticker
import matplotlib.font_manager as font_manager
##############################################################

lw = 0.5

SMALL_SIZE = 6
MEDIUM_SIZE = 7
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


CCC = sns.color_palette('Set2',100)
RED = sns.hls_palette(20,l=.5,s=1)[0]
PURPLE = sns.hls_palette(10,l=.45,s=1)[8]
BLUE = sns.hls_palette(15,l=.6,s=1)[9]
GREEN = sns.hls_palette(8,l=.4,s=.8)[3]
PINK = sns.hls_palette(8,l=.7,s=1)[7]
ORNG = sns.hls_palette(20,l=.5,s=1)[2] 
TURQ = sns.color_palette('husl',8)[4]

DARKGREEN = '#14b847cc'
BRIGHTBLUE = '#00ffffcc'

LIGHTBLUE = '#93d3fb'
LIGHTPURPLE = '#d2a0f9'
LIGHTRED = '#ff6c70'
LIGHTORNG = '#fffe87'
LIGHTORNG2 = '#fec47f'

DARKGRAY = '#666666cc'
LIGHTGRAY = '#b2b2b2ff'
sns.set(font_scale = 1.5)
gray = [0.4,0.4,0.4]; 
gray2 = [0.6,0.6,0.6]
colorss = [CCC[0],CCC[3],CCC[1],RED,CCC[4],CCC[2]]
colorsss = [TURQ,PINK,ORNG,RED,PURPLE,BLUE]
colorsF = [BLUE,PINK,ORNG,RED,PURPLE,GREEN]
colorsG = [gray2,RED,ORNG,BLUE,PURPLE,GREEN]
colors_smc = [BLUE,PURPLE,ORNG,RED]
my_color_palette = colors_smc

# Then, "ALWAYS use sans-serif fonts"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['font.sans-serif'] = "Arial"

#path = '/usr/share/fonts/msttcore/arial.ttf'
#prop = font_manager.FontProperties(fname=path)
#mpl.rcParams['font.family'] = prop.get_name()

sns.reset_orig()
fontdict = {'family': 'sans-serif',
             'color':  'k',
             'weight': 'normal',
             'size': 16,
           }

plt.rcParams['svg.fonttype'] = 'none'
#######################################################################################

