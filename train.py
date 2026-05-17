"""
Training loop for EQL quantum symbolic regression.

Trains one EQL model per quantum system, per noise level.
Saves best weights, training history, and prints loss at each epoch.

Usage:
    python train.py                        # train all systems, all noise levels
    python train.py --system qho           # train QHO only
    python train.py --system hydrogen      # train hydrogen only
    python train.py --system de_broglie    # train de Broglie only
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data.generate    import generate_all, NOISE_LEVELS
from models.eql_network import (
    build_qho_model,
    build_hydrogen_model,
    build_de_broglie_model,
    DataNormalizer,
    print_model_summary,
)

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS       = 800
BATCH_SIZE   = 64
LR           = 1e-3
L1_REG = 1e-4
PATIENCE     = 60      # early stopping patience
VAL_SPLIT    = 0.15     # 15% of data used for validation

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


# ── Train one model ───────────────────────────────────────────────────────────

def train_one(system, noise_level, X, y,
              epochs=EPOCHS, lr=LR, l1_reg=L1_REG,
              save_dir='checkpoints'):
    """
    Train one EQL model for a given system and noise level.

    Args:
        system      : 'qho', 'hydrogen', or 'de_broglie'
        noise_level : float, e.g. 0.0, 0.01, 0.05, 0.10
        X           : input array (num_samples, input_dim)
        y           : target array (num_samples,)
        epochs      : max training epochs
        lr          : learning rate
        l1_reg      : L1 regularisation strength
        save_dir    : directory to save weights and history

    Returns:
        model       : trained Keras model
        eql_layers  : list of EQLLayer instances
        history     : training history dict
        normalizer  : fitted DataNormalizer
    """
    tag      = f'{system}_noise{int(noise_level*100):02d}pct'
    save_path = os.path.join(save_dir, tag)
    os.makedirs(save_path, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Training : {system.upper()}  |  Noise: {noise_level*100:.0f}%')
    print(f'  True eq  : {SYSTEM_TRUE_EQ[system]}')
    print(f'{"="*60}')
    # ── Normalise data ────────────────────────────────────────────────────────
    # de Broglie spans 12 orders of magnitude — needs log scaling
    use_log    = (system == 'de_broglie')
    normalizer = DataNormalizer(use_log_X=use_log, use_log_y=use_log)
    X_norm, y_norm = normalizer.fit_transform(X, y.astype(np.float32))
    X_norm     = X_norm.astype(np.float32)
    y_norm     = y_norm.astype(np.float32)

    # de Broglie needs slower learning rate due to log-scale complexity
    actual_lr  = lr * 0.1 if system == 'de_broglie' else lr

    # ── Train / val split ─────────────────────────────────────────────────────
    n_val   = int(len(X_norm) * VAL_SPLIT)
    idx     = np.random.permutation(len(X_norm))
    val_idx = idx[:n_val]
    trn_idx = idx[n_val:]

    X_train, y_train = X_norm[trn_idx], y_norm[trn_idx]
    X_val,   y_val   = X_norm[val_idx], y_norm[val_idx]

    # ── Build model ───────────────────────────────────────────────────────────
    builder          = SYSTEM_BUILDERS[system]
    model, eql_layers = builder(l1_reg=l1_reg)

    if noise_level == 0.0:
        print_model_summary(model, eql_layers)

    # ── Compile ───────────────────────────────────────────────────────────────
    optimizer = keras.optimizers.Adam(learning_rate=lr,clipnorm=1.0,)
    model.compile(
        optimizer=optimizer,
        loss='huber',
        metrics=['mae'],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
            keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=25,
        min_lr=1e-7,
        verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, 'best_weights.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,   # we print our own summary below
    )

    # ── Print result ──────────────────────────────────────────────────────────
    best_val  = min(hist.history['val_loss'])
    best_ep   = hist.history['val_loss'].index(best_val) + 1
    final_mae = hist.history['val_mae'][-1]
    epochs_ran = len(hist.history['loss'])

    print(f'  Epochs ran   : {epochs_ran}  (early stop at {best_ep})')
    print(f'  Best val MSE : {best_val:.6f}')
    print(f'  Final val MAE: {final_mae:.6f}')

    # ── R² score on validation set ────────────────────────────────────────────
    y_pred = model.predict(X_val, verbose=0).flatten()
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - y_val.mean()) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-10)
    print(f'  Val R²       : {r2:.4f}')

    # ── Save history ──────────────────────────────────────────────────────────
    history = {
        'train_loss' : hist.history['loss'],
        'val_loss'   : hist.history['val_loss'],
        'train_mae'  : hist.history['mae'],
        'val_mae'    : hist.history['val_mae'],
        'best_val_loss': float(best_val),
        'best_epoch' : int(best_ep),
        'val_r2'     : float(r2),
        'noise_level': noise_level,
        'system'     : system,
    }
    with open(os.path.join(save_path, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, eql_layers, history, normalizer


# ── Train all systems × all noise levels ──────────────────────────────────────

def train_all(systems=None, noise_levels=None):
    """
    Train EQL models for all combinations of system and noise level.

    Returns:
        results : nested dict results[system][noise] = {
                    model, eql_layers, history, normalizer
                  }
    """
    if systems is None:
        systems = list(SYSTEM_BUILDERS.keys())
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    print('Generating datasets...')
    # de Broglie needs more samples due to 2D input complexity
    num_samples = 5000 if 'de_broglie' in systems else 2000
    all_data = generate_all(num_samples=num_samples, save=False)

    results = {}

    for system in systems:
        results[system] = {}
        for noise in noise_levels:
            X, y = all_data[system][noise]
            model, eql_layers, history, normalizer = train_one(
                system, noise, X, y
            )
            results[system][noise] = {
                'model'     : model,
                'eql_layers': eql_layers,
                'history'   : history,
                'normalizer': normalizer,
            }

    # ── Print summary table ───────────────────────────────────────────────────
    print(f'\n\n{"="*65}')
    print(f'  TRAINING COMPLETE — SUMMARY TABLE')
    print(f'{"="*65}')
    print(f'{"System":<14} {"Noise":>6} {"Best Val MSE":>13} '
          f'{"Val R²":>8} {"Best Epoch":>11}')
    print(f'{"─"*65}')

    for system in systems:
        for noise in noise_levels:
            h = results[system][noise]['history']
            print(f'{system:<14} {noise*100:>5.0f}%  '
                  f'{h["best_val_loss"]:>13.6f}  '
                  f'{h["val_r2"]:>8.4f}  '
                  f'{h["best_epoch"]:>11}')
    print(f'{"="*65}')

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train EQL symbolic regression models'
    )
    parser.add_argument(
        '--system',
        type=str,
        default='all',
        choices=['all', 'qho', 'hydrogen', 'de_broglie'],
        help='Which quantum system to train (default: all)',
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=None,
        help='Single noise level to train (default: all levels)',
    )
    args = parser.parse_args()

    systems = (
        list(SYSTEM_BUILDERS.keys())
        if args.system == 'all'
        else [args.system]
    )
    noise_levels = (
        NOISE_LEVELS
        if args.noise is None
        else [args.noise]
    )

    train_all(systems=systems, noise_levels=noise_levels)