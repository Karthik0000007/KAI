# Aegis Plugin Development Kit

Welcome to the Aegis Plugin Development Kit. This guide explains how to create plugins to extend Aegis's functionality.

## Overview

Aegis supports 4 types of plugins:
1. `HealthSignalExtractorPlugin` - Extracts custom health signals from user text
2. `ProactiveCheckPlugin` - Implements custom proactive interventions
3. `TTSBackendPlugin` - Adds support for new Text-To-Speech engines
4. `EmotionClassifierPlugin` - Implements custom emotion classification models

All plugins run inside a sandbox that prevents them from breaking the core system and ensures data privacy.

## Quick Start

1. Create a new python file in the `plugins/` directory.
2. Import the desired plugin ABC from `core.plugin_system`.
3. Create a class that inherits from the ABC and implements all abstract methods.
4. Add your plugin's configuration to `data/plugins.yaml`.

See `example_plugin.py` for a working implementation of a `HealthSignalExtractorPlugin`.

## Testing

Aegis provides a testing harness in `core.plugin_test_utils.py`. You can use this to write unit tests for your plugins without needing to run the full Aegis application.

```python
from core.plugin_test_utils import PluginTestHarness
from plugins.my_plugin import MyPlugin

def test_my_plugin():
    harness = PluginTestHarness(MyPlugin())
    
    # Validation tests
    harness.assert_plugin_valid()
    
    # Extractor tests
    signals = harness.run_extractor_test("I drank 8 glasses of water", "en")
    assert signals.get("water_glasses") == 8
```
