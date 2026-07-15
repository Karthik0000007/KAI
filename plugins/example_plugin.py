import re
from typing import Dict, Any
from core.plugin_system import HealthSignalExtractorPlugin

class WaterIntakeExtractor(HealthSignalExtractorPlugin):
    """
    Example plugin that extracts water intake signals from user text.
    Demonstrates how to implement the HealthSignalExtractorPlugin interface.
    """
    
    def __init__(self):
        self._enabled = True
        self.water_goal = 8

    @property
    def name(self) -> str:
        return "WaterIntakeExtractor"
        
    @property
    def version(self) -> str:
        return "1.0.0"
        
    @property
    def author(self) -> str:
        return "Aegis Team"
        
    @property
    def description(self) -> str:
        return "Extracts daily water intake (number of glasses) from user text."

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize with configuration."""
        self._enabled = config.get("track_water_intake", True)
        self.water_goal = config.get("water_goal_glasses", 8)
        return True

    def validate(self) -> bool:
        """Self-check."""
        return isinstance(self.water_goal, int) and self.water_goal > 0

    def extract_signals(self, text: str, language: str) -> Dict[str, Any]:
        """Extract water intake from text."""
        signals = {}
        if not self._enabled:
            return signals
            
        text_lower = text.lower()
        
        # English patterns
        if language == "en":
            match = re.search(r'(?:drank|had)\s+(\d+)\s+glasses\s+of\s+water', text_lower)
            if match:
                signals["water_glasses"] = int(match.group(1))
                
        # Japanese patterns
        elif language == "ja":
            match = re.search(r'水を?(\d+)(?:杯|グラス|コップ)\s*(?:飲んだ|のみました)', text_lower)
            if match:
                signals["water_glasses"] = int(match.group(1))
                
        # Check against goal if extracted
        if "water_glasses" in signals:
            signals["water_goal_met"] = signals["water_glasses"] >= self.water_goal
            
        return signals

    def shutdown(self) -> None:
        """Cleanup."""
        pass
