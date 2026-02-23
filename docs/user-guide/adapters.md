# Writing Adapters

Adapters wrap external objects — typically ML models — so they can be used with
Calibrax's benchmarking and profiling tools. Calibrax provides two adapter
base classes for different use cases.

## When to Use Adapters

Use an adapter when you need to benchmark an object that does not conform to
Calibrax's protocols. For example, a Flax NNX model has a `__call__` method
but not the `setup()` / `run_training()` / `run_evaluation()` interface that
`BenchmarkProtocol` expects.

## BenchmarkAdapter (ABC)

`BenchmarkAdapter` is an abstract base class for wrapping non-JAX targets such
as PyTorch models or plain Python objects:

```python
from calibrax.core.adapters import BenchmarkAdapter

class PyTorchAdapter(BenchmarkAdapter):
    @classmethod
    def can_adapt(cls, target: object) -> bool:
        # Return True if this adapter can wrap the given target
        return hasattr(target, "forward")

    @property
    def name(self) -> str:
        return type(self.target).__name__

# Usage
adapter = PyTorchAdapter(my_pytorch_model)
print(adapter.name)          # "MyModel"
print(adapter.target)        # the original model
```

Key points:

- Subclass `BenchmarkAdapter` and override `can_adapt()` to declare which
  objects your adapter supports
- Access the wrapped object via the `target` property
- The `name` property defaults to `target.name`, then `target.model_name`,
  falling back to `"unknown"`

## NNXBenchmarkAdapter

`NNXBenchmarkAdapter` inherits from `nnx.Module` (not `BenchmarkAdapter`),
making it compatible with JAX transformations like `nnx.jit`, `nnx.vmap`,
and `nnx.grad`:

```python
from flax import nnx
from calibrax.core.adapters import NNXBenchmarkAdapter

model = nnx.Linear(128, 64, rngs=nnx.Rngs(0))
adapter = NNXBenchmarkAdapter(model)
print(adapter.name)  # "unknown" (nnx.Linear has no .name attribute)
```

!!! tip "Using nnx.jit"

    `NNXBenchmarkAdapter` is intentionally minimal — subclasses add
    domain-specific methods like `predict()`. Because `nnx.jit` does not
    support bound methods, use the unbound method pattern:

    ```python
    class MyAdapter(NNXBenchmarkAdapter):
        def predict(self, x):
            return self.model(x)

    adapter = MyAdapter(model)

    # Correct: unbound method + instance
    result = nnx.jit(MyAdapter.predict)(adapter, x)

    # Incorrect — will fail
    result = nnx.jit(adapter.predict)(x)
    ```

`NNXBenchmarkAdapter.can_adapt()` returns `True` for any `nnx.Module` instance.

## AdapterRegistry

The `AdapterRegistry` manages adapter resolution. When you call `adapt()`,
it tries each registered adapter in priority order (most recently registered
first) and returns the first one whose `can_adapt()` returns `True`:

```python
from calibrax.core.adapters import adapt, register_adapter

# Register a custom adapter
register_adapter(PyTorchAdapter)

# adapt() tries adapters in reverse registration order
wrapped = adapt(my_model)
```

The default registry pre-registers `NNXBenchmarkAdapter`, so NNX models are
adapted automatically:

```python
from calibrax.core.adapters import adapt

wrapped = adapt(my_nnx_model)  # returns NNXBenchmarkAdapter
```

### Manual Registry

For isolated testing or custom resolution logic:

```python
from calibrax.core.adapters import AdapterRegistry

registry = AdapterRegistry()
registry.register(PyTorchAdapter)
registry.register(NNXBenchmarkAdapter)

wrapped = registry.adapt(model)
registry.reset()  # clear all registrations
```

## Custom Adapter Example

A complete adapter for a hypothetical framework:

```python
from calibrax.core.adapters import BenchmarkAdapter, register_adapter

class SklearnAdapter(BenchmarkAdapter):
    """Adapter for scikit-learn estimators."""

    @classmethod
    def can_adapt(cls, target: object) -> bool:
        return hasattr(target, "fit") and hasattr(target, "predict")

    @property
    def name(self) -> str:
        return type(self.target).__name__

    def predict(self, x):
        return self.target.predict(x)

    def fit(self, x, y):
        return self.target.fit(x, y)

# Register globally
register_adapter(SklearnAdapter)
```

## Best Practices

- Use `NNXBenchmarkAdapter` for Flax NNX models — it preserves JIT compatibility
- Use `BenchmarkAdapter` for everything else — PyTorch, scikit-learn, custom objects
- Register adapters early (e.g., at module load time) so `adapt()` can resolve
  them when needed
- Override `can_adapt()` with precise checks to avoid false matches
- Keep adapters thin — delegate to the wrapped object rather than reimplementing
  logic

## Next Steps

<div class="grid cards" markdown>

-   :material-timer:{ .lg .middle } **Profiling**

    ---

    Profile adapted models with timing, resources, and GPU analysis

    [:octicons-arrow-right-24: Profiling](profiling.md)

-   :material-book-open:{ .lg .middle } **Core Concepts**

    ---

    Understand protocols and the data model that adapters connect to

    [:octicons-arrow-right-24: Concepts](../getting-started/concepts.md)

</div>
