from typing import Dict, Any, List, Optional
from core.plugin_system import (
    AegisPlugin, 
    HealthSignalExtractorPlugin, 
    ProactiveCheckPlugin,
    TTSBackendPlugin,
    EmotionClassifierPlugin
)
from core.models import ProactiveAlert, EmotionResult

class PluginTestHarness:
    """Utility class for testing Aegis plugins."""
    
    def __init__(self, plugin: AegisPlugin, config: Optional[Dict[str, Any]] = None):
        self.plugin = plugin
        self.config = config or {}
        
        # Auto-initialize
        self.is_initialized = self.plugin.initialize(self.config)
        
    def assert_plugin_valid(self):
        """Assert that the plugin meets basic requirements."""
        assert self.plugin.name, "Plugin must have a name"
        assert self.plugin.version, "Plugin must have a version"
        assert self.plugin.author, "Plugin must have an author"
        assert self.plugin.description, "Plugin must have a description"
        assert self.is_initialized, "Plugin failed to initialize"
        assert self.plugin.validate(), "Plugin self-validation failed"
        assert len(self.plugin.get_capabilities()) > 0, "Plugin must implement at least one capability"

    def run_extractor_test(self, text: str, language: str) -> Dict[str, Any]:
        """Test a HealthSignalExtractorPlugin."""
        if not isinstance(self.plugin, HealthSignalExtractorPlugin):
            raise TypeError("Plugin must be a HealthSignalExtractorPlugin")
            
        return self.plugin.extract_signals(text, language)

    def run_proactive_check_test(self, mock_db_state: Dict[str, Any]) -> List[ProactiveAlert]:
        """Test a ProactiveCheckPlugin."""
        if not isinstance(self.plugin, ProactiveCheckPlugin):
            raise TypeError("Plugin must be a ProactiveCheckPlugin")
            
        return self.plugin.run_check(mock_db_state, self.config)

    def run_tts_test(self, text: str, language: str, tone_mode: str, filepath: str) -> bool:
        """Test a TTSBackendPlugin."""
        if not isinstance(self.plugin, TTSBackendPlugin):
            raise TypeError("Plugin must be a TTSBackendPlugin")
            
        assert language in self.plugin.supported_languages(), f"Language {language} not supported"
        return self.plugin.synthesize(text, language, tone_mode, filepath)
        
    def run_emotion_test(self, features: Dict[str, Any], transcript: Optional[str] = None) -> EmotionResult:
        """Test an EmotionClassifierPlugin."""
        if not isinstance(self.plugin, EmotionClassifierPlugin):
            raise TypeError("Plugin must be a EmotionClassifierPlugin")
            
        return self.plugin.classify(features, transcript)
        
    def cleanup(self):
        """Shutdown the plugin."""
        self.plugin.shutdown()
