"""
Generate a dependency graph showing how the trainers/ folder
calls into each of the other folders in the TERSE project.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── colour palette ──────────────────────────────────────────────
C_TRAINER  = '#3B7DD8'   # blue
C_CONFIG   = '#E8A838'   # amber
C_DATA     = '#50B86C'   # green
C_ALGO     = '#D94A6E'   # rose
C_MODEL    = '#8E44AD'   # purple
C_UTIL     = '#6C7A89'   # grey
C_TEXT     = '#FFFFFF'

# ── helper: draw a rounded box ──────────────────────────────────
def draw_box(x, y, w, h, color, title, items, fontsize=9.5):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.18",
                         facecolor=color, edgecolor='white',
                         linewidth=2.5, alpha=0.92, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.30, title,
            ha='center', va='top', fontsize=fontsize + 3,
            fontweight='bold', color=C_TEXT, zorder=3)
    for i, item in enumerate(items):
        ax.text(x + w/2, y + h - 0.75 - i * 0.36, item,
                ha='center', va='top', fontsize=fontsize,
                color=C_TEXT, family='monospace', zorder=3)

# ── helper: draw arrow with label ───────────────────────────────
def draw_arrow(x1, y1, x2, y2, label='', color='#555', lw=2.0,
               label_x=None, label_y=None, rad=0.05):
    ax.annotate('',
                xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>',
                                color=color, lw=lw,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=1)
    if label:
        lx = label_x if label_x else (x1 + x2) / 2
        ly = label_y if label_y else (y1 + y2) / 2
        ax.text(lx, ly, label, ha='center', va='center',
                fontsize=8.5, color=color, fontstyle='italic',
                fontweight='medium',
                bbox=dict(boxstyle='round,pad=0.2', fc='#FAFAFA',
                          ec=color, alpha=0.92, linewidth=0.8),
                zorder=4)

# ═══════════════════════════════════════════════════════════════
#  BOXES  (x, y, w, h)
# ═══════════════════════════════════════════════════════════════

# ── trainers/ (top-centre) ──────────────────────────────────────
tw, th = 9, 3.2
tx, ty = (18 - tw) / 2, 9.2
draw_box(tx, ty, tw, th, C_TRAINER,
         'trainers/',
         ['train.py  --  entry point, arg parsing, scenario loop',
          'abstract_trainer.py  --  load data, train model, evaluate',
          'test.py  --  load checkpoint, test on target domain'])

# ── configs/ (bottom-left) ──────────────────────────────────────
cw, ch = 4.8, 2.8
cx, cy = 0.4, 4.0
draw_box(cx, cy, cw, ch, C_CONFIG,
         'configs/',
         ['data_model_configs.py',
          '  get_dataset_class()',
          'hparams.py',
          '  get_hparams_class()'])

# ── dataloader/ (bottom-centre-left) ────────────────────────────
dw, dh = 4.5, 2.8
dx, dy = 6.0, 4.0
draw_box(dx, dy, dw, dh, C_DATA,
         'dataloader/',
         ['dataloader.py',
          '  data_generator()',
          '  Load_Dataset class'])

# ── algorithms/ (bottom-right) ──────────────────────────────────
aw, ah = 4.8, 2.8
axx, ay = 12.8, 4.0
draw_box(axx, ay, aw, ah, C_ALGO,
         'algorithms/',
         ['algorithms.py',
          '  get_algorithm_class()',
          '  TERSE.pretrain() / .update()'])

# ── models/ (far bottom-right) ──────────────────────────────────
mw, mh = 5.2, 2.5
mx, my = 12.4, 0.5
draw_box(mx, my, mw, mh, C_MODEL,
         'models/',
         ['models.py  --  get_backbone_class()',
          'loss.py  --  CrossEntropyLabelSmooth, EntropyLoss'])

# ── utils.py (far bottom-left) ──────────────────────────────────
uw, uh = 4.8, 2.5
ux, uy = 0.4, 0.5
draw_box(ux, uy, uw, uh, C_UTIL,
         'utils.py  (root)',
         ['fix_randomness()',
          'starting_logs()',
          'AverageMeter'])

# ═══════════════════════════════════════════════════════════════
#  ARROWS  (from trainers → each folder)
# ═══════════════════════════════════════════════════════════════

# trainers → configs  (abstract_trainer imports get_dataset_class, get_hparams_class)
draw_arrow(tx + 1.0, ty,
           cx + cw/2, cy + ch,
           'get_dataset_class()\nget_hparams_class()',
           C_CONFIG, label_x=3.2, label_y=8.2, rad=0.08)

# trainers → dataloader  (abstract_trainer imports data_generator)
draw_arrow(tx + tw/2, ty,
           dx + dw/2, dy + dh,
           'data_generator()',
           C_DATA, label_x=9.0, label_y=8.2, rad=0.0)

# trainers → algorithms  (train_model calls get_algorithm_class, pretrain, update)
draw_arrow(tx + tw - 1.0, ty,
           axx + aw/2, ay + ah,
           'get_algorithm_class()\npretrain() / update()',
           C_ALGO, label_x=14.5, label_y=8.2, rad=-0.08)

# trainers → models  (abstract_trainer imports get_backbone_class)
draw_arrow(tx + tw - 0.3, ty,
           mx + mw/2, my + mh,
           'get_backbone_class()',
           C_MODEL, label_x=16.5, label_y=7.2, rad=-0.22)

# trainers → utils  (train.py imports fix_randomness, starting_logs, AverageMeter)
draw_arrow(tx + 0.3, ty,
           ux + uw/2, uy + uh,
           'fix_randomness()\nstarting_logs()\nAverageMeter',
           C_UTIL, label_x=1.2, label_y=7.2, rad=0.22)

# algorithms → models  (TERSE internally uses backbone, Classifier, Imputer, etc.)
draw_arrow(axx + aw/2, ay,
           mx + mw/2, my + mh,
           'backbone, Classifier,\nTemporal_Imputer, GraphRecover',
           C_MODEL, label_x=15.8, label_y=3.6, rad=0.0)

# ── title ───────────────────────────────────────────────────────
ax.text(9, 12.6, 'TERSE  --  How trainers/ calls other folders',
        ha='center', va='center', fontsize=18, fontweight='bold',
        color='#2C3E50')

# ── legend ──────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_TRAINER, label='trainers/'),
    mpatches.Patch(color=C_CONFIG,  label='configs/'),
    mpatches.Patch(color=C_DATA,    label='dataloader/'),
    mpatches.Patch(color=C_ALGO,    label='algorithms/'),
    mpatches.Patch(color=C_MODEL,   label='models/'),
    mpatches.Patch(color=C_UTIL,    label='utils.py'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=6,
          fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
plt.savefig('trainers_dependency_graph.png', dpi=180, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Saved to trainers_dependency_graph.png")
