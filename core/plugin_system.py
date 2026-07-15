import os
import sys
import time
import uuid
import logging
import importlib.util
from typing import Dict, List, Any, Optional, Callable, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import copy
import concurrent.futures

from core.models import EmotionResult, ProactiveAlert

logger = logging.getLogger("aegis.plugins")

# ─── Plugin API ─────────────────────────────────────────────────────────────

class AegisPlugin(ABC):
    """Base class for all Aegis plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def version(self) -> str:
        pass
        
    @property
    @abstractmethod
    def author(self) -> str:
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        return True

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass

    def validate(self) -> bool:
        """Self-check capability."""
        return True

    def get_capabilities(self) -> List[str]:
        """Return list of capabilities (interface names) this plugin implements."""
        caps = []
        if isinstance(self, HealthSignalExtractorPlugin):
            caps.append("extractor")
        if isinstance(self, ProactiveCheckPlugin):
            caps.append("proactive")
        if isinstance(self, TTSBackendPlugin):
            caps.append("tts")
        if isinstance(self, EmotionClassifierPlugin):
            caps.append("emotion")
        return caps


class HealthSignalExtractorPlugin(AegisPlugin):
    """Plugin for custom health signal extraction."""
    @abstractmethod
    def extract_signals(self, text: str, language: str) -> Dict[str, Any]:
        """Extract signals. Must return a dict matching Aegis signal schema."""
        pass


class ProactiveCheckPlugin(AegisPlugin):
    """Plugin for custom proactive interventions."""
    @abstractmethod
    def run_check(self, db_state: Dict[str, Any], config: Dict[str, Any]) -> List[ProactiveAlert]:
        """Run analysis on read-only DB state and return alerts."""
        pass


class TTSBackendPlugin(AegisPlugin):
    """Plugin for custom Text-To-Speech."""
    @abstractmethod
    def synthesize(self, text: str, language: str, tone_mode: str, filepath: str) -> bool:
        """Synthesize text to audio file."""
        pass
        
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        pass


class EmotionClassifierPlugin(AegisPlugin):
    """Plugin for custom emotion classification."""
    @abstractmethod
    def classify(self, features: Dict[str, Any], transcript: Optional[str] = None) -> EmotionResult:
        """Classify emotion from audio features and transcript."""
        pass


# ─── Plugin Sandboxing ──────────────────────────────────────────────────────

class PluginTimeoutError(Exception):
    pass

class PluginSandbox:
    """Provides isolated execution environment for plugins."""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a plugin function with timeout and deep copied arguments to prevent
        mutation of internal state.
        """
        # Deep copy arguments to prevent plugins from modifying internal state
        safe_args = copy.deepcopy(args)
        safe_kwargs = copy.deepcopy(kwargs)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *safe_args, **safe_kwargs)
            try:
                result = future.result(timeout=self.timeout_seconds)
                return result
            except concurrent.futures.TimeoutError:
                logger.error(f"Plugin execution timed out after {self.timeout_seconds}s")
                raise PluginTimeoutError(f"Execution timed out after {self.timeout_seconds}s")
            except Exception as e:
                logger.error(f"Plugin execution failed: {e}")
                raise


# ─── Plugin Loader & Registry ───────────────────────────────────────────────

@dataclass
class PluginMetadata:
    name: str
    version: str
    author: str
    module_path: str
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[AegisPlugin] = None


class PluginLoader:
    """Discovers, validates, and loads plugins."""
    
    def __init__(self, plugins_dir: Path, config: Dict[str, Any]):
        self.plugins_dir = plugins_dir
        self.config = config
        self.plugins: Dict[str, PluginMetadata] = {}

    def discover_plugins(self) -> List[PluginMetadata]:
        """Scan directory for valid plugin modules."""
        discovered = []
        if not self.plugins_dir.exists():
            logger.warning(f"Plugins directory {self.plugins_dir} does not exist.")
            return discovered

        for file_path in self.plugins_dir.glob("*.py"):
            if file_path.name.startswith("__") or file_path.name.startswith("test_"):
                continue
            
            # Create basic metadata (will be populated fully on load)
            plugin_name = file_path.stem
            is_enabled = self.config.get("installed", {}).get(plugin_name, {}).get("enabled", True)
            
            meta = PluginMetadata(
                name=plugin_name,
                version="unknown",
                author="unknown",
                module_path=str(file_path),
                enabled=is_enabled,
                config=self.config.get("installed", {}).get(plugin_name, {}).get("config", {})
            )
            discovered.append(meta)
            
        return discovered

    def load_plugin(self, meta: PluginMetadata) -> Optional[AegisPlugin]:
        """Load and instantiate a plugin from its module."""
        if not meta.enabled:
            logger.info(f"Plugin {meta.name} is disabled. Skipping.")
            return None

        try:
            # Use importlib to load the module dynamically
            spec = importlib.util.spec_from_file_location(f"plugins.{meta.name}", meta.module_path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load plugin {meta.name} from {meta.module_path}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the AegisPlugin subclass
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, AegisPlugin) and attr is not AegisPlugin and attr.__module__ == module.__name__:
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                logger.error(f"No AegisPlugin subclass found in {meta.name}")
                return None
                
            # Instantiate and validate
            instance = plugin_class()
            meta.name = instance.name
            meta.version = instance.version
            meta.author = instance.author
            
            if not self.validate_plugin(instance):
                logger.error(f"Plugin {meta.name} failed validation.")
                return None
                
            meta.instance = instance
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin {meta.name}: {e}")
            return None

    def validate_plugin(self, plugin: AegisPlugin) -> bool:
        """Verify plugin implements required methods and passes self-check."""
        try:
            return plugin.validate()
        except Exception as e:
            logger.error(f"Plugin validation threw exception: {e}")
            return False


class PluginRegistry:
    """Manages active plugins organized by capability."""
    
    def __init__(self, sandbox_timeout: int = 30):
        self._plugins: Dict[str, PluginMetadata] = {}
        self._by_capability: Dict[str, List[AegisPlugin]] = {
            "extractor": [],
            "proactive": [],
            "tts": [],
            "emotion": []
        }
        self.sandbox = PluginSandbox(timeout_seconds=sandbox_timeout)

    def register(self, meta: PluginMetadata) -> bool:
        """Register a loaded plugin."""
        if meta.instance is None:
            return False
            
        # Check if already registered
        if meta.name in self._plugins:
            logger.warning(f"Plugin {meta.name} already registered.")
            return False

        # Initialize
        try:
            if not meta.instance.initialize(meta.config):
                logger.error(f"Plugin {meta.name} failed to initialize.")
                return False
        except Exception as e:
            logger.error(f"Plugin {meta.name} initialization error: {e}")
            return False

        # Register
        self._plugins[meta.name] = meta
        
        # Categorize by capability
        for cap in meta.instance.get_capabilities():
            if cap in self._by_capability:
                self._by_capability[cap].append(meta.instance)
                
        logger.info(f"Registered plugin: {meta.name} v{meta.version} ({', '.join(meta.instance.get_capabilities())})")
        return True

    def unregister(self, name: str) -> bool:
        """Unregister and shutdown a plugin."""
        if name not in self._plugins:
            return False
            
        meta = self._plugins[name]
        if meta.instance:
            try:
                meta.instance.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down plugin {name}: {e}")
                
            # Remove from capabilities mapping
            for cap in meta.instance.get_capabilities():
                if cap in self._by_capability and meta.instance in self._by_capability[cap]:
                    self._by_capability[cap].remove(meta.instance)
                    
        del self._plugins[name]
        logger.info(f"Unregistered plugin: {name}")
        return True

    def get_plugins(self, capability: str) -> List[AegisPlugin]:
        """Get all plugins implementing a specific capability."""
        return self._by_capability.get(capability, [])

    def execute_extractor(self, plugin: HealthSignalExtractorPlugin, text: str, language: str) -> Dict[str, Any]:
        """Execute an extractor plugin safely."""
        return self.sandbox.execute(plugin.extract_signals, text, language)

    def execute_proactive(self, plugin: ProactiveCheckPlugin, db_state: Dict[str, Any], config: Dict[str, Any]) -> List[ProactiveAlert]:
        """Execute a proactive plugin safely."""
        return self.sandbox.execute(plugin.run_check, db_state, config)
