# Gradient Clipping and Regularization Semantics

This note documents the current behavior in `packages/engine/src/network.ts`.
It is descriptive only: it does not propose or require a change to clipping or
regularization semantics.

## Current Update Order

For each batch, the engine accumulates raw gradients during backpropagation and
then applies updates in two passes:

1. Average all weight and bias gradients by the batch/output normalization
   count.
2. Compute one global gradient norm from those averaged weight and bias
   gradients.
3. If `gradientClip` is a positive number and the norm exceeds it, compute a
   scalar clip factor `clip / norm`; otherwise use `1`.
4. Run the selected optimizer. The clipped data gradient is passed into the
   optimizer, and weight regularization is added inside the optimizer update.

Important details:

- The global norm includes weight gradients and bias gradients.
- The global norm is computed before regularization terms are added.
- Biases are not regularized.
- `regularizationRate` contributes only when `regularization` is `l1` or `l2`.
- For L1/L2, the regularization term is added to weight updates after applying
  the clip scale to the data gradient.

In shorthand, current weight-gradient handling is:

```text
dataGrad = averagedBatchGradient
scaledDataGrad = dataGrad * clipScale

if l2:
  optimizerInput = scaledDataGrad + regularizationRate * weight
else if l1:
  optimizerInput = scaledDataGrad + regularizationRate * sign(weight)
else:
  optimizerInput = scaledDataGrad
```

## Comparison With Classical L2

Classical L2 regularization adds a penalty term to the objective, commonly:

```text
lossWithPenalty = dataLoss + 0.5 * lambda * sum(weight^2)
```

The corresponding weight gradient contribution is:

```text
dataGrad + lambda * weight
```

The playground's `l2` mode matches that coupled-gradient form at the point where
the optimizer receives its weight gradient. For plain SGD with no gradient
clipping, this is equivalent to multiplicative weight decay:

```text
weight = weight - lr * (dataGrad + lambda * weight)
       = (1 - lr * lambda) * weight - lr * dataGrad
```

With the current global-norm clipping, however, the clipping norm is based on
`dataGrad` only. The `lambda * weight` term is added after the clip factor is
computed, so the regularization contribution is not itself clipped. This means
current clipping limits the batch-gradient contribution, not the full
regularized objective gradient.

With momentum or Adam, coupled L2 also enters the optimizer state: the
regularization contribution is included in momentum buffers or Adam moments.
That is standard coupled L2 behavior, but it is not the same as decoupled weight
decay.

## Comparison With AdamW-Style Decoupled Weight Decay

AdamW-style decoupled weight decay treats weight decay as a separate parameter
update rather than as part of the gradient passed into Adam's moment estimates.
Conceptually:

```text
adamStep = Adam(dataGrad)
weight = weight - adamStep - lr * weightDecay * weight
```

The playground's current Adam path does not do this. In `adam` mode with `l2`,
the engine adds `regularizationRate * weight` to the gradient before updating
Adam's first and second moments:

```text
g = scaledDataGrad + regularizationRate * weight
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g * g
weight = weight - lr * mHat / (sqrt(vHat) + eps)
```

Consequences:

- L2 regularization is adaptive under Adam because it is normalized by Adam's
  second-moment estimate.
- The L2 term affects both first and second moments.
- The update is not equivalent to AdamW decoupled weight decay.
- Gradient clipping still applies only to the data-gradient portion before the
  L2 term is added.

If AdamW-style behavior is added in the future, it should be introduced as an
explicit semantic change, with tests that distinguish coupled L2 from decoupled
weight decay and preserve the existing global-norm clipping contract unless the
contract is intentionally revised.
