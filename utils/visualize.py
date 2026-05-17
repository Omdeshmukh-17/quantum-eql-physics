"""
Visualisation utilities for EQL quantum symbolic regression.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix import path so data and models modules are found
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs('results/figures', exist_ok=True)

COLORS = {
    'qho'       : '#7F77DD',
    'hydrogen'  : '#1D9E75',
    'de_broglie': '#EF9F27',
}
LABELS = {
    'qho'       : 'Quantum Harmonic Oscillator',
    'hydrogen'  : 'Hydrogen Atom',
    'de_broglie': 'de Broglie Relation',
}
NOISE_LABELS = ['0%', '1%', '5%', '10%']
NOISE_VALS   = [0.0, 0.01, 0.05, 0.10]

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linewidth'   : 0.6,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
})


def plot_noise_robustness(results_path='results/evaluation.json',
                          save_path='results/figures/noise_robustness.png'):
    with open(results_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    systems    = ['qho', 'hydrogen', 'de_broglie']
    linestyles = ['-', '--', ':']

    for system, ls in zip(systems, linestyles):
        sys_r  = sorted(
            [r for r in results if r['system'] == system],
            key=lambda x: x['noise_level']
        )
        noises = [r['noise_level'] * 100 for r in sys_r]
        r2s    = [r['r2'] for r in sys_r]

        ax.plot(noises, r2s,
                color=COLORS[system], linestyle=ls,
                linewidth=2.5, marker='o', markersize=8,
                label=LABELS[system])

        ax.annotate(f'{r2s[-1]:.4f}',
                    xy=(noises[-1], r2s[-1]),
                    xytext=(8, 0),
                    textcoords='offset points',
                    fontsize=9, color=COLORS[system])

    ax.set_xlabel('Gaussian noise level (%)', fontsize=12)
    ax.set_ylabel('R² score', fontsize=12)
    ax.set_title(
        'Noise robustness of EQL symbolic regression\n'
        'across three quantum physics systems',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xticks([0, 1, 5, 10])
    ax.set_ylim(0.93, 1.002)
    ax.axhline(1.0, color='gray', linestyle='--',
               linewidth=0.8, alpha=0.5, label='Perfect recovery')
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved noise robustness plot → {save_path}')


def plot_r2_bars(results_path='results/evaluation.json',
                 save_path='results/figures/r2_comparison.png'):
    with open(results_path) as f:
        results = json.load(f)

    systems = ['qho', 'hydrogen', 'de_broglie']
    x       = np.arange(len(NOISE_VALS))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, system in enumerate(systems):
        sys_r = sorted(
            [r for r in results if r['system'] == system],
            key=lambda r: r['noise_level']
        )
        r2s  = [r['r2'] for r in sys_r]
        bars = ax.bar(
            x + i * width, r2s, width,
            label=LABELS[system],
            color=COLORS[system],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5,
        )
        for bar, v in zip(bars, r2s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'{v:.3f}',
                ha='center', va='bottom',
                fontsize=8, color=COLORS[system],
                fontweight='bold'
            )

    ax.set_xlabel('Gaussian noise level', fontsize=12)
    ax.set_ylabel('R² score', fontsize=12)
    ax.set_title(
        'EQL model performance across systems and noise levels',
        fontsize=13, fontweight='bold', pad=12
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(NOISE_LABELS)
    ax.set_ylim(0.92, 1.015)
    ax.axhline(1.0, color='gray', linestyle='--',
               linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved R² bar chart → {save_path}')


def plot_parity(save_path='results/figures/parity_plots.png'):
    from data.generate import (
        generate_qho, generate_hydrogen, generate_de_broglie
    )
    from models.eql_network import (
        build_qho_model, build_hydrogen_model,
        build_de_broglie_model, DataNormalizer
    )

    configs = [
        ('qho',        generate_qho,
         build_qho_model,        False),
        ('hydrogen',   generate_hydrogen,
         build_hydrogen_model,   False),
        ('de_broglie', generate_de_broglie,
         build_de_broglie_model, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (system, gen_fn, build_fn, use_log) in zip(axes, configs):
        X, y     = gen_fn(num_samples=500, noise_level=0.0, seed=99)
        norm     = DataNormalizer(use_log_X=use_log, use_log_y=use_log)
        X_n, y_n = norm.fit_transform(X, y.astype(np.float32))
        X_n      = X_n.astype(np.float32)

        weights_path = f'checkpoints/{system}_noise00pct/best_weights.keras'
        if not os.path.exists(weights_path):
            ax.text(0.5, 0.5, 'No weights found',
                    ha='center', transform=ax.transAxes)
            continue

        model, _ = build_fn(l1_reg=1e-4)
        _        = model(X_n[:2], training=False)
        model.load_weights(weights_path)

        y_pred_n = model.predict(X_n, verbose=0).flatten()

        ax.scatter(y_n, y_pred_n,
                   alpha=0.4, s=15,
                   color=COLORS[system],
                   edgecolors='none')

        lo = min(y_n.min(), y_pred_n.min()) - 0.02
        hi = max(y_n.max(), y_pred_n.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], 'k--',
                linewidth=1.2, alpha=0.6)

        ss_res = np.sum((y_n - y_pred_n) ** 2)
        ss_tot = np.sum((y_n - y_n.mean()) ** 2)
        r2     = 1 - ss_res / (ss_tot + 1e-10)

        ax.text(0.05, 0.93, f'R² = {r2:.4f}',
                transform=ax.transAxes,
                fontsize=11, fontweight='bold',
                color=COLORS[system],
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white',
                          edgecolor=COLORS[system],
                          alpha=0.8))

        ax.set_xlabel('Actual (normalised)', fontsize=10)
        ax.set_ylabel('Predicted (normalised)', fontsize=10)
        ax.set_title(LABELS[system], fontsize=11,
                     fontweight='bold', color=COLORS[system])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    fig.suptitle(
        'Parity plots — predicted vs actual (clean data, 0% noise)',
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved parity plots → {save_path}')


def plot_training_curves(system='hydrogen', noise_level=0.0,
                         save_path=None):
    if save_path is None:
        save_path = f'results/figures/training_curves_{system}.png'

    tag          = f'{system}_noise{int(noise_level*100):02d}pct'
    history_path = f'checkpoints/{tag}/history.json'

    if not os.path.exists(history_path):
        print(f'No history found for {tag}')
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(epochs, history['train_loss'],
            color=COLORS[system], linewidth=1.5,
            label='Train', alpha=0.9)
    ax.plot(epochs, history['val_loss'],
            color=COLORS[system], linewidth=1.5,
            linestyle='--', label='Val', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss (Huber)', fontsize=11)
    ax.set_title(f'Training loss — {LABELS[system]}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(epochs, history['train_mae'],
            color=COLORS[system], linewidth=1.5,
            label='Train MAE', alpha=0.9)
    ax.plot(epochs, history['val_mae'],
            color=COLORS[system], linewidth=1.5,
            linestyle='--', label='Val MAE', alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title(f'Training MAE — {LABELS[system]}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved training curves → {save_path}')


def plot_robustness_heatmap(results_path='results/evaluation.json',
                            save_path='results/figures/robustness_heatmap.png'):
    with open(results_path) as f:
        results = json.load(f)

    systems = ['qho', 'hydrogen', 'de_broglie']
    noises  = [0.01, 0.05, 0.10]
    data    = np.zeros((len(systems), len(noises)))

    for i, system in enumerate(systems):
        baseline = next(
            r['r2'] for r in results
            if r['system'] == system and r['noise_level'] == 0.0
        )
        for j, noise in enumerate(noises):
            r2 = next(
                r['r2'] for r in results
                if r['system'] == system
                and abs(r['noise_level'] - noise) < 1e-6
            )
            data[i, j] = round(baseline - r2, 4)

    fig, ax = plt.subplots(figsize=(7, 4))
    im      = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(noises)))
    ax.set_xticklabels(['1% noise', '5% noise', '10% noise'])
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels([LABELS[s] for s in systems])

    for i in range(len(systems)):
        for j in range(len(noises)):
            ax.text(j, i, f'{data[i,j]:.4f}',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='black' if data[i, j] < 0.03 else 'white')

    plt.colorbar(im, ax=ax, label='R² drop from clean baseline')
    ax.set_title(
        'Noise robustness heatmap — R² degradation\n'
        '(lower is better — smaller drop = more robust)',
        fontsize=12, fontweight='bold', pad=10
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved robustness heatmap → {save_path}')


def generate_all_figures():
    print('Generating all figures...\n')

    plot_noise_robustness()
    plot_r2_bars()
    plot_parity()
    plot_training_curves(system='hydrogen',   noise_level=0.0)
    plot_training_curves(system='qho',        noise_level=0.0)
    plot_training_curves(system='de_broglie', noise_level=0.0)
    plot_robustness_heatmap()

    print('\nAll figures saved to results/figures/')
    print('Files:')
    for f in sorted(os.listdir('results/figures')):
        size = os.path.getsize(f'results/figures/{f}')
        print(f'  {f:<45} ({size/1024:.1f} KB)')


if __name__ == '__main__':
    generate_all_figures()