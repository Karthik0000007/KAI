import pytest
import os
import time
from pathlib import Path
from typing import Dict, Any, List

from core.plugin_system import (
    AegisPlugin,
    HealthSignalExtractorPlugin,
    ProactiveCheckPlugin,
    TTSBackendPlugin,
    EmotionClassifierPlugin,
    PluginLoader,
    PluginMetadata,
    PluginRegistry,
    PluginSandbox,
    PluginTimeoutError
)
from core.models import ProactiveAlert, EmotionResult
from core.plugin_test_utils import PluginTestHarness

# ─── Mocks & Fixtures ───────────────────────────────────────────────────────

class ValidExtractorPlugin(HealthSignalExtractorPlugin):
    @property
    def name(self) -> str: return "TestExtractor"
    @property
    def version(self) -> str: return "1.0"
    @property
    def author(self) -> str: return "Test"
    @property
    def description(self) -> str: return "Test"
    
    def extract_signals(self, text: str, language: str) -> Dict[str, Any]:
        return {"test_signal": True}

class SlowPlugin(HealthSignalExtractorPlugin):
    @property
    def name(self) -> str: return "SlowPlugin"
    @property
    def version(self) -> str: return "1.0"
    @property
    def author(self) -> str: return "Test"
    @property
    def description(self) -> str: return "Test"
    
    def extract_signals(self, text: str, language: str) -> Dict[str, Any]:
        time.sleep(2.0)
        return {"done": True}

class MutatingPlugin(HealthSignalExtractorPlugin):
    @property
    def name(self) -> str: return "Mutator"
    @property
    def version(self) -> str: return "1.0"
    @property
    def author(self) -> str: return "Test"
    @property
    def description(self) -> str: return "Test"
    
    def extract_signals(self, text: str, language: str) -> Dict[str, Any]:
        # Try to mutate an argument
        if isinstance(text, list):
            text.append("hacked")
        return {"done": True}

@pytest.fixture
def temp_plugin_dir(tmp_path):
    # Create a temporary plugin directory
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    
    # Write a dummy plugin file
    plugin_content = """
from typing import Dict, Any
from core.plugin_system import HealthSignalExtractorPlugin

class DummyExtractor(HealthSignalExtractorPlugin):
    @property
    def name(self): return "DummyExtractor"
    @property
    def version(self): return "1.0"
    @property
    def author(self): return "Author"
    @property
    def description(self): return "Description"
    
    def extract_signals(self, text: str, language: str) -> Dict[str, Any]:
        return {"dummy": True}
"""
    (plugins_dir / "dummy_plugin.py").write_text(plugin_content)
    
    # Write an invalid plugin file
    invalid_content = """
class InvalidPlugin:
    pass
"""
    (plugins_dir / "invalid_plugin.py").write_text(invalid_content)
    
    return plugins_dir

# ─── API Tests ──────────────────────────────────────────────────────────────

def test_plugin_abc_cannot_instantiate():
    with pytest.raises(TypeError):
        AegisPlugin()
        
    with pytest.raises(TypeError):
        HealthSignalExtractorPlugin()

def test_plugin_capabilities():
    plugin = ValidExtractorPlugin()
    caps = plugin.get_capabilities()
    assert caps == ["extractor"]

# ─── Sandbox Tests ──────────────────────────────────────────────────────────

def test_sandbox_timeout():
    sandbox = PluginSandbox(timeout_seconds=1)
    plugin = SlowPlugin()
    
    with pytest.raises(PluginTimeoutError):
        sandbox.execute(plugin.extract_signals, "test", "en")

def test_sandbox_prevents_mutation():
    sandbox = PluginSandbox(timeout_seconds=5)
    plugin = MutatingPlugin()
    
    # Try to mutate a list passed as an argument
    arg_list = ["safe"]
    sandbox.execute(plugin.extract_signals, arg_list, "en")
    
    # Assert original argument is unmodified
    assert arg_list == ["safe"]

def test_sandbox_successful_execution():
    sandbox = PluginSandbox(timeout_seconds=5)
    plugin = ValidExtractorPlugin()
    
    result = sandbox.execute(plugin.extract_signals, "test", "en")
    assert result == {"test_signal": True}

# ─── Loader Tests ───────────────────────────────────────────────────────────

def test_discover_plugins(temp_plugin_dir):
    loader = PluginLoader(temp_plugin_dir, {})
    discovered = loader.discover_plugins()
    
    assert len(discovered) == 2
    names = [meta.name for meta in discovered]
    assert "dummy_plugin" in names
    assert "invalid_plugin" in names

def test_load_valid_plugin(temp_plugin_dir):
    loader = PluginLoader(temp_plugin_dir, {})
    discovered = loader.discover_plugins()
    
    dummy_meta = next(m for m in discovered if m.name == "dummy_plugin")
    plugin = loader.load_plugin(dummy_meta)
    
    assert plugin is not None
    assert plugin.name == "DummyExtractor"
    assert "extractor" in plugin.get_capabilities()

def test_load_invalid_plugin(temp_plugin_dir):
    loader = PluginLoader(temp_plugin_dir, {})
    discovered = loader.discover_plugins()
    
    invalid_meta = next(m for m in discovered if m.name == "invalid_plugin")
    plugin = loader.load_plugin(invalid_meta)
    
    assert plugin is None

def test_load_disabled_plugin(temp_plugin_dir):
    config = {"installed": {"dummy_plugin": {"enabled": False}}}
    loader = PluginLoader(temp_plugin_dir, config)
    discovered = loader.discover_plugins()
    
    dummy_meta = next(m for m in discovered if m.name == "dummy_plugin")
    assert not dummy_meta.enabled
    
    plugin = loader.load_plugin(dummy_meta)
    assert plugin is None

# ─── Registry Tests ─────────────────────────────────────────────────────────

def test_registry_register_unregister():
    registry = PluginRegistry()
    plugin = ValidExtractorPlugin()
    meta = PluginMetadata(name=plugin.name, version="1.0", author="Test", module_path="", instance=plugin)
    
    assert registry.register(meta)
    assert len(registry.get_plugins("extractor")) == 1
    
    assert not registry.register(meta) # Duplicate registration
    
    assert registry.unregister(plugin.name)
    assert len(registry.get_plugins("extractor")) == 0

def test_registry_execution():
    registry = PluginRegistry(sandbox_timeout=5)
    plugin = ValidExtractorPlugin()
    
    result = registry.execute_extractor(plugin, "test", "en")
    assert result == {"test_signal": True}

# ─── Test Utils ─────────────────────────────────────────────────────────────

def test_plugin_test_harness():
    plugin = ValidExtractorPlugin()
    harness = PluginTestHarness(plugin)
    
    harness.assert_plugin_valid()
    
    result = harness.run_extractor_test("text", "en")
    assert result == {"test_signal": True}
    
    harness.cleanup()
