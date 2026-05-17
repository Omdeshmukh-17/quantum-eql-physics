"""
Full Equation Learner (EQL) Network — built from EQLLayer blocks.

Architecture:
  Input → EQLLayer × num_layers → Linear output head → scalar prediction

The final layer is always a plain Dense(1) with no activation —
this is the regression output (predicted energy, wavelength, etc.)

Three pre-configured models for each quantum system:
  build_qho_model()        — for quantum harmonic oscillator
  build_hydrogen_model()   — for hydrogen atom
  build_de_broglie_model() — for de Broglie relation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.eql_layer import EQLLayer, SymbolicExtractor

# ── Generic EQL Network builder ───────────────────────────────────────────────

def build_eql_network(input_dim, num_layers=2,
                      num_unary=2, num_binary=1,
                      funcs=None, l1_reg=1e-3,
                      name='eql_network'):
    """
    Build a full EQL network as a Keras Functional model.

    Args:
        input_dim  : number of input features
        num_layers : number of EQL layers (2–3 is optimal for simple physics)
        num_unary  : unary units per function per layer
        num_binary : binary (multiply) units per layer
        funcs      : list of functions to include in library
        l1_reg     : L1 regularisation strength
        name       : model name

    Returns:
        model      : compiled Keras model
        eql_layers : list of EQLLayer instances (for symbolic extraction)
    """
    if funcs is None:
        funcs = ['id', 'sin', 'cos', 'exp', 'inv', 'square']

    inputs     = keras.Input(shape=(input_dim,), name='input')
    x          = inputs
    eql_layers = []

    for i in range(num_layers):
        layer = EQLLayer(
            num_unary=num_unary,
            num_binary=num_binary,
            funcs=funcs,
            l1_reg=l1_reg,
            name=f'eql_layer_{i}',
        )
        x = layer(x)
        eql_layers.append(layer)

    # Final linear output — no activation for regression
    output = keras.layers.Dense(
        1, use_bias=True,
        kernel_regularizer=keras.regularizers.L1(l1_reg),
        name='output'
    )(x)

    # Squeeze to (batch,) for scalar regression
    output = keras.layers.Flatten(name='flatten')(output)

    model = keras.Model(inputs=inputs, outputs=output, name=name)
    return model, eql_layers


# ── System-specific model builders ────────────────────────────────────────────

def build_qho_model(l1_reg=1e-3):
    """
    EQL model for Quantum Harmonic Oscillator.

    Input  : [n, omega]   — 2 features
    Target : E_n = hbar * omega * (n + 0.5)

    The true equation needs: identity, multiply.
    We include full library so model must discover this.
    """
    model, eql_layers = build_eql_network(
        input_dim=2,
        num_layers=2,
        num_unary=2,
        num_binary=1,
        funcs=['id', 'sin', 'cos', 'exp', 'inv', 'square'],
        l1_reg=l1_reg,
        name='eql_qho',
    )
    return model, eql_layers


def build_hydrogen_model(l1_reg=1e-3):
    """
    EQL model for Hydrogen Atom Energy Levels.

    Input  : [n]           — 1 feature
    Target : E_n = -13.6 / n^2

    The true equation needs: inv, square.
    """
    model, eql_layers = build_eql_network(
        input_dim=1,
        num_layers=2,
        num_unary=2,
        num_binary=1,
        funcs=['id', 'sin', 'cos', 'exp', 'inv', 'square'],
        l1_reg=l1_reg,
        name='eql_hydrogen',
    )
    return model, eql_layers


def build_de_broglie_model(l1_reg=1e-3):
    """
    EQL model for de Broglie Relation.

    Input  : [mass, velocity]  — 2 features
    Target : lambda = h / (mass * velocity)

    The true equation needs: multiply, inv.
    """
    model, eql_layers = build_eql_network(
        input_dim=2,
        num_layers=2,
        num_unary=2,
        num_binary=1,
        funcs=['id', 'sin', 'cos', 'exp', 'inv', 'square'],
        l1_reg=l1_reg,
        name='eql_de_broglie',
    )
    return model, eql_layers


# ── Normalisation helpers ─────────────────────────────────────────────────────

class DataNormalizer:
    """
    Normaliser for physics data.
    Supports standard min-max and log-scale normalisation.
    Log scale is critical for de Broglie data which spans
    many orders of magnitude.
    """

    def __init__(self, use_log_X=False, use_log_y=False):
        """
        Args:
            use_log_X : apply log10 to inputs before normalising
            use_log_y : apply log10 to targets before normalising
        """
        self.use_log_X = use_log_X
        self.use_log_y = use_log_y
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None

    def fit(self, X, y):
        X_ = np.log10(np.abs(X) + 1e-40) if self.use_log_X else X
        y_ = np.log10(np.abs(y) + 1e-40) if self.use_log_y else y
        self.X_min = X_.min(axis=0)
        self.X_max = X_.max(axis=0)
        self.y_min = y_.min()
        self.y_max = y_.max()
        return self

    def transform_X(self, X):
        X_ = np.log10(np.abs(X) + 1e-40) if self.use_log_X else X
        denom = self.X_max - self.X_min
        denom = denom + (denom == 0) * 1e-8
        return (X_ - self.X_min) / denom

    def transform_y(self, y):
        y_ = np.log10(np.abs(y) + 1e-40) if self.use_log_y else y
        denom = self.y_max - self.y_min
        denom = denom + (denom == 0) * 1e-8
        return (y_ - self.y_min) / denom

    def inverse_transform_y(self, y_norm):
        y_ = y_norm * (self.y_max - self.y_min) + self.y_min
        return (10 ** y_) if self.use_log_y else y_

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform_X(X), self.transform_y(y)

# ── Model summary utility ─────────────────────────────────────────────────────

def print_model_summary(model, eql_layers):
    """Print model architecture and parameter count."""
    print(f'\nModel: {model.name}')
    print(f'{"─"*50}')
    total = sum(
        v.numpy().size for v in model.trainable_variables
    )
    print(f'Total trainable parameters: {total:,}')
    print(f'EQL layers: {len(eql_layers)}')
    for i, layer in enumerate(eql_layers):
        out_dim = layer.output_dim()
        print(f'  Layer {i}: functions={layer.funcs}, '
              f'num_unary={layer.num_unary}, '
              f'num_binary={layer.num_binary}, '
              f'output_dim={out_dim}')
    print(f'{"─"*50}\n')


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np

    print('Testing all three EQL network configurations...\n')

    configs = [
        ('QHO',        build_qho_model,        2),
        ('Hydrogen',   build_hydrogen_model,    1),
        ('de Broglie', build_de_broglie_model,  2),
    ]

    for name, builder, input_dim in configs:
        model, eql_layers = builder(l1_reg=1e-3)
        print_model_summary(model, eql_layers)

        # Test forward pass
        x_test = tf.random.normal((8, input_dim))
        y_pred = model(x_test, training=False)
        print(f'  {name} forward pass: '
              f'input={x_test.shape} → output={y_pred.shape}')

        # Test normaliser
        X_dummy = np.random.randn(100, input_dim).astype(np.float32)
        y_dummy = np.random.randn(100).astype(np.float32)
        norm    = DataNormalizer()
        X_n, y_n = norm.fit_transform(X_dummy, y_dummy)
        print(f'  Normaliser: X range [{X_n.min():.2f}, {X_n.max():.2f}], '
              f'y range [{y_n.min():.2f}, {y_n.max():.2f}]')
        print()

    print('All network tests passed.')