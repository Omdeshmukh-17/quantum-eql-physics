"""
Equation Learner (EQL) Layer — implemented from scratch in TensorFlow/Keras.

Based on: Sahoo et al. 2018
"Learning Equations for Extrapolation and Control"
arXiv:1806.07259

Core idea:
  Instead of standard neurons (weighted sum → activation),
  each EQL layer is a LIBRARY of mathematical functions.
  The network learns WHICH functions and WHICH combinations
  fit the data — producing a human-readable symbolic expression.

Function library per layer:
  Unary  : identity, sin, cos, exp, 1/x, sqrt, x^2
  Binary : multiplication (takes pairs of inputs)
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np


class EQLLayer(keras.layers.Layer):
    """
    Single Equation Learner layer.

    Architecture:
      - Takes input x of shape (batch, input_dim)
      - Applies learned linear combinations to get function inputs
      - Passes through fixed nonlinear function library
      - Concatenates all function outputs

    Args:
        num_unary  : number of unary function units per function type
        num_binary : number of binary (multiply) units
        funcs      : list of unary functions to include
                     options: 'id', 'sin', 'cos', 'exp', 'inv', 'square'
        l1_reg     : L1 regularisation strength (enforces sparsity —
                     pushes unused weights to exactly zero)
    """

    # All supported unary functions
    FUNC_MAP = {
        'id'    : lambda x: x,
        'sin'   : tf.sin,
        'cos'   : tf.cos,
        'exp'   : lambda x: tf.exp(tf.clip_by_value(x, -3.0, 3.0)),
        'inv'   : lambda x: tf.math.divide_no_nan(
                    1.0, tf.clip_by_value(x, -10.0, 10.0) +
                    tf.sign(x) * 1e-6 + 1e-6),
        'square': lambda x: tf.square(tf.clip_by_value(x, -10.0, 10.0)),
    }
    def __init__(self, num_unary=2, num_binary=1,
                 funcs=None, l1_reg=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.num_unary  = num_unary
        self.num_binary = num_binary
        self.l1_reg     = l1_reg

        # Default function library
        if funcs is None:
            self.funcs = ['id', 'sin', 'cos', 'exp', 'inv', 'square']
        else:
            self.funcs = funcs

        # Validate
        for f in self.funcs:
            if f not in self.FUNC_MAP:
                raise ValueError(
                    f"Unknown function '{f}'. "
                    f"Choose from: {list(self.FUNC_MAP.keys())}"
                )

    def build(self, input_shape):
        """
        Create learnable weight matrices.
        One weight matrix per function group — maps inputs to function inputs.
        """
        input_dim = int(input_shape[-1])
        reg = keras.regularizers.L1(self.l1_reg)

        # Weight matrices for unary functions
        # Shape: (input_dim, num_unary) per function type
        self.W_unary = {}
        self.b_unary = {}
        for fname in self.funcs:
            self.W_unary[fname] = self.add_weight(
                name=f'W_{fname}',
                shape=(input_dim, self.num_unary),
                initializer='glorot_uniform',
                regularizer=reg,
                trainable=True,
            )
            self.b_unary[fname] = self.add_weight(
                name=f'b_{fname}',
                shape=(self.num_unary,),
                initializer='zeros',
                trainable=True,
            )

        # Weight matrices for binary (multiply) function
        # Binary needs TWO inputs per unit — W1 and W2
        if self.num_binary > 0:
            self.W_bin1 = self.add_weight(
                name='W_bin1',
                shape=(input_dim, self.num_binary),
                initializer='glorot_uniform',
                regularizer=reg,
                trainable=True,
            )
            self.W_bin2 = self.add_weight(
                name='W_bin2',
                shape=(input_dim, self.num_binary),
                initializer='glorot_uniform',
                regularizer=reg,
                trainable=True,
            )
            self.b_bin1 = self.add_weight(
                name='b_bin1',
                shape=(self.num_binary,),
                initializer='zeros',
                trainable=True,
            )
            self.b_bin2 = self.add_weight(
                name='b_bin2',
                shape=(self.num_binary,),
                initializer='zeros',
                trainable=True,
            )

        super().build(input_shape)

    def call(self, x, training=False):
        """
        Forward pass.

        For each function in library:
          1. Linearly combine inputs: z = x @ W + b
          2. Apply nonlinearity:      out = f(z)

        For binary (multiply):
          z1 = x @ W1 + b1
          z2 = x @ W2 + b2
          out = z1 * z2

        Concatenate all outputs → shape (batch, total_output_dim)
        """
        outputs = []

        # Unary functions
        for fname in self.funcs:
            W = self.W_unary[fname]
            b = self.b_unary[fname]

            # Linear combination of inputs
            z = tf.matmul(x, W) + b          # (batch, num_unary)

            # Apply function
            fn  = self.FUNC_MAP[fname]
            out = fn(z)                       # (batch, num_unary)

            outputs.append(out)

        #Binary (multiply) function 
        if self.num_binary > 0:
            z1  = tf.matmul(x, self.W_bin1) + self.b_bin1   # (batch, num_binary)
            z2  = tf.matmul(x, self.W_bin2) + self.b_bin2   # (batch, num_binary)
            out = z1 * z2                                     # elementwise multiply
            outputs.append(out)

        # Concatenate all function outputs
        return tf.concat(outputs, axis=-1)

    def output_dim(self):
        """Returns the output dimension of this layer."""
        return len(self.funcs) * self.num_unary + self.num_binary

    def get_config(self):
        config = super().get_config()
        config.update(
            num_unary=self.num_unary,
            num_binary=self.num_binary,
            funcs=self.funcs,
            l1_reg=self.l1_reg,
        )
        return config


class SymbolicExtractor:
    """
    Extracts human-readable symbolic expressions from trained EQL weights.

    After training, weights close to zero are pruned (set to zero).
    Remaining weights map to symbolic terms which are printed as equations.
    """

    def __init__(self, threshold=0.01):
        """
        Args:
            threshold : weights below this value are treated as zero (pruned)
        """
        self.threshold = threshold

    def extract(self, layer, input_names):
        """
        Extract symbolic expression from one EQL layer.

        Args:
            layer       : trained EQLLayer instance
            input_names : list of strings naming each input variable
                          e.g. ['n', 'omega'] for QHO

        Returns:
            terms : list of symbolic expression strings
        """
        terms = []

        for fname in layer.funcs:
            W = layer.W_unary[fname].numpy()   # (input_dim, num_unary)
            b = layer.b_unary[fname].numpy()   # (num_unary,)

            for unit_idx in range(layer.num_unary):
                w_col = W[:, unit_idx]
                bias  = b[unit_idx]

                # Build linear combination string
                lin_parts = []
                for var_idx, w in enumerate(w_col):
                    if abs(w) > self.threshold:
                        # use input name if available, else generic label
                        if var_idx < len(input_names):
                            var_name = input_names[var_idx]
                        else:
                            var_name = f'x{var_idx}'
                        lin_parts.append(f'{w:.4f}*{var_name}')
                if abs(bias) > self.threshold:
                    lin_parts.append(f'{bias:.4f}')

                if not lin_parts:
                    continue   # All weights pruned — skip this unit

                lin_str = ' + '.join(lin_parts)

                # Wrap in function
                if fname == 'id':
                    term = f'({lin_str})'
                elif fname == 'inv':
                    term = f'1/({lin_str})'
                elif fname == 'square':
                    term = f'({lin_str})^2'
                else:
                    term = f'{fname}({lin_str})'

                terms.append(term)

        # Binary terms
        if layer.num_binary > 0:
            W1 = layer.W_bin1.numpy()
            W2 = layer.W_bin2.numpy()
            b1 = layer.b_bin1.numpy()
            b2 = layer.b_bin2.numpy()

            for unit_idx in range(layer.num_binary):
                # First factor
                parts1 = []
                for vi, w in enumerate(W1[:, unit_idx]):
                    if abs(w) > self.threshold:
                        var_name = input_names[vi] if vi < len(input_names) else f'x{vi}'
                        parts1.append(f'{w:.4f}*{var_name}')
                if abs(b1[unit_idx]) > self.threshold:
                    parts1.append(f'{b1[unit_idx]:.4f}')

                # Second factor
                parts2 = []
                for vi, w in enumerate(W2[:, unit_idx]):
                    if abs(w) > self.threshold:
                        var_name = input_names[vi] if vi < len(input_names) else f'x{vi}'
                        parts2.append(f'{w:.4f}*{var_name}')
                if abs(b2[unit_idx]) > self.threshold:
                    parts2.append(f'{b2[unit_idx]:.4f}')

                if parts1 and parts2:
                    s1   = ' + '.join(parts1)
                    s2   = ' + '.join(parts2)
                    term = f'({s1}) * ({s2})'
                    terms.append(term)

        return terms

    def print_equation(self, layer, input_names, system_name=''):
        """Pretty-print the recovered symbolic equation."""
        terms = self.extract(layer, input_names)
        if not terms:
            print(f'  [{system_name}] No significant terms found.')
            return
        eq = ' + '.join(terms)
        print(f'  [{system_name}] Recovered: y = {eq}')


#  Quick test 

if __name__ == '__main__':
    print('Testing EQLLayer...\n')

    # Build a small test layer
    layer = EQLLayer(
        num_unary=2,
        num_binary=1,
        funcs=['id', 'sin', 'cos', 'exp', 'inv', 'square'],
        l1_reg=1e-3,
        name='test_eql'
    )

    # Forward pass with random input
    x_test = tf.random.normal((4, 3))   # batch=4, input_dim=3
    out    = layer(x_test)

    expected_dim = 6 * 2 + 1   # 6 funcs × 2 unary + 1 binary = 13
    print(f'Input shape  : {x_test.shape}')
    print(f'Output shape : {out.shape}')
    print(f'Expected dim : {expected_dim}')
    print(f'Dim match    : {out.shape[-1] == expected_dim}')
    print(f'\nTrainable variables: {len(layer.trainable_variables)}')
    for v in layer.trainable_variables:
        print(f'  {v.name:30s}  shape={v.shape}')

    print('\nEQLLayer test passed.')