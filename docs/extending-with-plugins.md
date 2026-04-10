# Extending Atropos with Plugins

Atropos now supports a plugin registry designed around two extension points:

1. **Environments** (Gym-style environment registry)
2. **Server backends** (Transformers-style backend/model registry)

This allows third-party packages to add environments and inference servers without
changing the Atropos core repository.

## Core API

```python
from atroposlib.plugins import PluginRegistry, register_builtin_servers

registry = PluginRegistry()
register_builtin_servers(registry)  # openai, vllm, sglang

# Load third-party plugins from installed packages
registry.load_entry_points()

env = registry.create_environment("third_party/gsm8k-example")
server = registry.create_server("third_party/openai-compatible")
```

### Built-in entry point groups

- `atropos.plugins`: bundle-style plugins that expose `register(registry)`
- `atropos.environments`: direct environment factories
- `atropos.servers`: direct server backend factories

## Creating a pip-installable plugin package

A minimal package layout:

```text
my-plugin/
  pyproject.toml
  src/my_plugin/__init__.py
```

### `pyproject.toml`

```toml
[project.entry-points."atropos.plugins"]
my_plugin = "my_plugin:register"
```

### `src/my_plugin/__init__.py`

```python
from atroposlib.plugins import PluginRegistry


def create_env():
    ...


def create_server(**kwargs):
    ...


def register(registry: PluginRegistry) -> None:
    registry.register_environment("my-org/my-env", create_env)
    registry.register_server("my-org/my-server", create_server)
```

## Example plugin in this repository

See the complete example package at:

- `examples/plugins/atropos-gsm8k-plugin/`

You can install it locally with:

```bash
pip install -e examples/plugins/atropos-gsm8k-plugin
```

Then load it:

```python
from atroposlib.plugins import PluginRegistry

registry = PluginRegistry()
registry.load_entry_points()
print(sorted(registry.environments))
print(sorted(registry.servers))
```

## Design notes

- **Dynamic loading**: `importlib.metadata.entry_points()` discovers installed plugins.
- **Third-party support**: any pip-installable package can publish Atropos entry points.
- **Server decoupling**: OpenAI, vLLM, and SGLang backends are registered as factories,
  so runtime server implementations can live outside core.
