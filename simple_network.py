import jax.numpy as jnp


# ---- Hardcoded integer weights for 2-3-1 ReLU net ----
# First layer: 2 -> 3
W1 = jnp.array([
    [ 8., 12., 20.],
    [12., 18., 30.]
])  # shape (2, 3)

b1 = jnp.array([
    -12., -18., -30.
])  # shape (3,)

# Second layer: 3 -> 1
W2 = jnp.array([
    [ 1.],
    [-2.],
    [ 4.]
])  # shape (3, 1)

b2 = jnp.array([0.])  # shape (1,)


def forward(x: jnp.ndarray) -> jnp.ndarray:
    """
    x: (batch, 2) array of [hw, exam] in [0, 1]
    returns: (batch, 1) raw outputs y
    """
    # Layer 1: affine + ReLU
    h_pre = x @ W1 + b1           # (batch, 3)
    h = jnp.maximum(h_pre, 0.0)   # ReLU

    # Layer 2: affine
    y = h @ W2 + b2               # (batch, 1)
    return y


def main():
    # Print weights and biases
    print("W1 (2x3):\n", W1)
    print("b1 (3,):\n", b1)
    print("W2 (3x1):\n", W2)
    print("b2 (1,):\n", b2)

    # ---- Example batch of students: [hw, exam] in [0, 1] ----
    X = jnp.array([
        [0.20, 0.20],  # low hw, low exam  -> likely fail
        [0.30, 0.90],  # decent hw, strong exam -> pass
        [0.80, 0.70],  # good both -> pass
        [0.50, 0.40],  # middling -> likely fail
        [0.70, 0.80],  # strong -> pass
    ])

    # Forward pass
    y = forward(X)              # raw outputs
    passed = (y > 0.0)          # boolean: y > 0 => pass

    print("\nInputs (hw, exam):\n", X)
    print("\nRaw outputs y:\n", y)
    print("\nPredicted pass? (y > 0):\n", passed)

    # Optional: show the original grade rule score 0.4*hw + 0.6*exam
    scores = 0.4 * X[:, 0] + 0.6 * X[:, 1]
    print("\nOriginal score 0.4*hw + 0.6*exam:\n", scores)


if __name__ == "__main__":
    main()

