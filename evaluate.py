"""
Evaluation and symbolic extraction for trained EQL models.

For each quantum system:
  1. Load trained weights
  2. Evaluate predictive accuracy (R², RMSE, MAE)
  3. Extract symbolic expression from learned weights
  4. Compare recovered equation to ground truth
  5. Run noise robustness analysis across all noise levels

Usage:
    python evaluate.py                    # evaluate all systems
    python evaluate.py --system qho       # evaluate QHO only
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

from data.generate import generate_all, NOISE_LEVELS
from models.eql_network import (
    build_qho_model,
    build_hydrogen_model,
    build_de_broglie_model,
    DataNormalizer,
)
from models.eql_layer import SymbolicExtractor

# ── Constants ─────────────────────────────────────────────────────────────────
SYSTEM_BUILDERS = {
    'qho'       : build_qho_model,
    'hydrogen'  : build_hydrogen_model,
    'de_broglie': build_de_broglie_model,
}

SYSTEM_INPUT_NAMES = {
    'qho'       : ['n', 'omega'],
    'hydrogen'  : ['n'],
    'de_broglie': ['mass', 'velocity'],
}

SYSTEM_TRUE_EQ = {
    'qho'       : 'E = hbar * omega * (n + 0.5)',
    'hydrogen'  : 'E = -13.6 / n^2',
    'de_broglie': 'lambda = h / (mass * velocity)',
}

SYSTEM_UNITS = {
    'qho'       : 'Joules',
    'hydrogen'  : 'eV',
    'de_broglie': 'metres',
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Compute R², RMSE, MAE on numpy arrays."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-10)
    rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae    = np.mean(np.abs(y_true - y_pred))
    return {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae)}


# ── Load and evaluate one model ───────────────────────────────────────────────

def evaluate_one(system, noise_level, X, y,
                 checkpoint_dir='checkpoints',
                 threshold=0.05):
    """
    Load trained weights for one system+noise combination,
    evaluate metrics, and extract symbolic expression.

    Args:
        system         : 'qho', 'hydrogen', or 'de_broglie'
        noise_level    : float noise level
        X              : input array
        y              : target array
        checkpoint_dir : directory containing saved weights
        threshold      : weight pruning threshold for symbolic extraction

    Returns:
        result dict with metrics and symbolic terms
    """
    tag        = f'{system}_noise{int(noise_level*100):02d}pct'
    weights_path = os.path.join(
        checkpoint_dir, tag, 'best_weights.keras'
    )

    if not os.path.exists(weights_path):
        print(f'  [SKIP] No weights found for {tag}')
        return None

    # ── Normalise ─────────────────────────────────────────────────────────────
    use_log    = (system == 'de_broglie')
    normalizer = DataNormalizer(use_log_X=use_log, use_log_y=use_log)
    X_norm, y_norm = normalizer.fit_transform(
        X, y.astype(np.float32)
    )
    X_norm = X_norm.astype(np.float32)
    y_norm = y_norm.astype(np.float32)

    # ── Build and load model ──────────────────────────────────────────────────
    builder          = SYSTEM_BUILDERS[system]
    model, eql_layers = builder(l1_reg=1e-4)

    # Warm up model to build weights before loading
    _ = model(X_norm[:2], training=False)
    model.load_weights(weights_path)

    # ── Predict ───────────────────────────────────────────────────────────────
    y_pred_norm = model.predict(X_norm, verbose=0).flatten()
    metrics     = compute_metrics(y_norm, y_pred_norm)

    # ── Symbolic extraction ───────────────────────────────────────────────────
    extractor    = SymbolicExtractor(threshold=threshold)
    input_names  = SYSTEM_INPUT_NAMES[system]
    symbolic_terms = []

    for i, layer in enumerate(eql_layers):
        terms = extractor.extract(layer, input_names)
        symbolic_terms.append({
            f'layer_{i}': terms
        })

    return {
        'system'        : system,
        'noise_level'   : noise_level,
        'metrics'       : metrics,
        'symbolic_terms': symbolic_terms,
        'true_equation' : SYSTEM_TRUE_EQ[system],
    }


# ── Evaluate all systems ──────────────────────────────────────────────────────

def evaluate_all(systems=None, noise_levels=None,
                 save_path='results/evaluation.json'):
    """
    Evaluate all system + noise combinations.
    Saves results to JSON and prints summary table.
    """
    if systems is None:
        systems = list(SYSTEM_BUILDERS.keys())
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    os.makedirs('results', exist_ok=True)

    print('Generating evaluation datasets...')
    all_data = generate_all(num_samples=2000, save=False)

    all_results = []

    for system in systems:
        print(f'\n{"="*60}')
        print(f'  Evaluating: {system.upper()}')
        print(f'  True equation: {SYSTEM_TRUE_EQ[system]}')
        print(f'{"="*60}')

        for noise in noise_levels:
            X, y = all_data[system][noise]
            print(f'\n  Noise: {noise*100:.0f}%')

            result = evaluate_one(system, noise, X, y)
            if result is None:
                continue

            m = result['metrics']
            print(f'    R²   : {m["r2"]:.4f}')
            print(f'    RMSE : {m["rmse"]:.6f}')
            print(f'    MAE  : {m["mae"]:.6f}')

            # Print symbolic terms
            print(f'    Symbolic terms (layer 0):')
            terms = result['symbolic_terms'][0].get('layer_0', [])
            if terms:
                for t in terms[:5]:   # show top 5
                    print(f'      {t}')
            else:
                print('      (all weights pruned below threshold)')

            all_results.append(result)

    # ── Save results ──────────────────────────────────────────────────────────
    # Convert to JSON-serialisable format
    serialisable = []
    for r in all_results:
        serialisable.append({
            'system'       : r['system'],
            'noise_level'  : r['noise_level'],
            'r2'           : round(r['metrics']['r2'],   4),
            'rmse'         : round(r['metrics']['rmse'], 6),
            'mae'          : round(r['metrics']['mae'],  6),
            'true_equation': r['true_equation'],
        })

    with open(save_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f'\nResults saved to {save_path}')

    # ── Print summary table ───────────────────────────────────────────────────
    print(f'\n\n{"="*65}')
    print(f'  EVALUATION SUMMARY')
    print(f'{"="*65}')
    print(f'{"System":<14} {"Noise":>6}  {"R²":>8}  {"RMSE":>10}  {"MAE":>10}')
    print(f'{"─"*65}')

    for r in serialisable:
        print(f'{r["system"]:<14} '
              f'{r["noise_level"]*100:>5.0f}%  '
              f'{r["r2"]:>8.4f}  '
              f'{r["rmse"]:>10.6f}  '
              f'{r["mae"]:>10.6f}')

    print(f'{"="*65}')

    # ── Noise robustness summary ──────────────────────────────────────────────
    print(f'\n  NOISE ROBUSTNESS — R² DROP FROM CLEAN BASELINE')
    print(f'{"─"*50}')

    for system in systems:
        sys_results = [r for r in serialisable if r['system'] == system]
        if not sys_results:
            continue
        baseline = next(
            (r['r2'] for r in sys_results if r['noise_level'] == 0.0),
            None
        )
        if baseline is None:
            continue
        print(f'\n  {system.upper()}  (baseline R² = {baseline:.4f})')
        for r in sys_results:
            if r['noise_level'] == 0.0:
                continue
            drop = baseline - r['r2']
            bar  = '█' * int(drop * 500)
            print(f'    {r["noise_level"]*100:>4.0f}% noise  '
                  f'R²={r["r2"]:.4f}  '
                  f'drop={drop:.4f}  {bar}')

    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained EQL symbolic regression models'
    )
    parser.add_argument(
        '--system',
        type=str,
        default='all',
        choices=['all', 'qho', 'hydrogen', 'de_broglie'],
    )
    args = parser.parse_args()

    systems = (
        list(SYSTEM_BUILDERS.keys())
        if args.system == 'all'
        else [args.system]
    )

    evaluate_all(systems=systems)
    