"""
style.py
--------
One figure style for every plot of the repository, matching the manuscript figures:
boxed axes, major and minor ticks pointing inward on the bottom and left axes only, no grid,
Arial-type bold labels, frameless legends, thick lines.

    from gouai_train import style
    style.apply()            # once, before creating figures
    ...
    style.finish(fig)        # before saving: frames, minor ticks, titles -> panel labels inside the frame
    style.save(fig, path)    # writes path.png and path.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, NullLocator  # noqa: E402

LW_AX, LW_LINE = 1.6, 2.0
# Pb-Pb green circles, Au-Au gold squares (as in the paper); new vs v6 models
SYSTEM_STYLE = {'2.76_PbPb': dict(marker='o', color='#008300'), '0.2_AuAu': dict(marker='s', color='#eda100')}
NEW, V6, TREND = '#2a78d6', '#eb6834', '#898781'
FOLD_SHADES = ['#0d366b', '#1c5cab', '#2a78d6', '#5598e7', '#86b6ef']


def _font():
    for d in ('/usr/share/fonts/liberation-sans', '/usr/share/fonts/truetype/liberation'):
        for f in Path(d).glob('LiberationSans-*.ttf'):
            font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans'):
        if name in names:
            return name
    return 'sans-serif'


def apply(size=11):
    font = _font()
    plt.rcParams.update({
        'font.family': font, 'font.size': size, 'font.weight': 'bold',
        'axes.labelweight': 'bold', 'axes.labelsize': size + 2, 'axes.titlesize': size + 1, 'axes.titleweight': 'bold',
        'axes.linewidth': LW_AX, 'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'axes.grid': False,
        'axes.spines.top': True, 'axes.spines.right': True,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': False, 'ytick.right': False,
        'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.major.width': LW_AX, 'ytick.major.width': LW_AX, 'xtick.minor.width': 1.0, 'ytick.minor.width': 1.0,
        'xtick.labelsize': size, 'ytick.labelsize': size, 'xtick.color': 'black', 'ytick.color': 'black',
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'lines.linewidth': LW_LINE, 'legend.frameon': False, 'legend.fontsize': size - 1,
        'mathtext.default': 'regular', 'figure.facecolor': 'white', 'savefig.dpi': 200})


def panel(a, text, x=0.035, y=0.965):
    a.text(x, y, text, transform=a.transAxes, ha='left', va='top', fontweight='bold',
           fontsize=plt.rcParams['font.size'] + 1, zorder=10)


def finish(fig):
    for a in fig.axes:
        a.grid(False)
        for s in a.spines.values():
            s.set_visible(True); s.set_linewidth(LW_AX); s.set_color('black')
        a.tick_params(which='both', direction='in', top=False, right=False, colors='black')
        for axis, scale in ((a.xaxis, a.get_xscale()), (a.yaxis, a.get_yscale())):
            if scale == 'linear' and not isinstance(axis.get_minor_locator(), NullLocator) \
                    and not isinstance(axis.get_major_formatter(), matplotlib.ticker.FixedFormatter):
                axis.set_minor_locator(AutoMinorLocator())
        t = a.get_title()
        if t:
            a.set_title('')
            panel(a, t)
    fig.tight_layout()


def save(fig, path):
    finish(fig)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{path}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('wrote', f'{path}.png')
