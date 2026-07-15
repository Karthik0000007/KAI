import pytest
import os
import yaml
from unittest.mock import patch
from pathlib import Path

from core.config import ConfigManager
import setup_wizard

import shutil

@pytest.fixture
def clean_prefs():
    # Setup - use the standard config path
    cm = ConfigManager()
    path = cm.config_path
    if path.exists():
        shutil.copy2(path, path.with_suffix('.yaml.bak'))
    yield
    # Teardown
    if path.with_suffix('.yaml.bak').exists():
        shutil.copy2(path.with_suffix('.yaml.bak'), path)
        path.with_suffix('.yaml.bak').unlink()

@patch("builtins.input")
def test_setup_wizard_preferences(mock_input, clean_prefs):
    # Simulate user entering choices:
    # Language: 3 (es)
    # DP Epsilon: 0.5
    # Retention days: 30
    mock_input.side_effect = ["3", "0.5", "30"]
    
    # Run config
    setup_wizard.configure_preferences()
    
    cm = ConfigManager()
    assert cm.config_path.exists()
    
    with open(cm.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    assert config["global"]["language"] == "es"
    assert config["privacy"]["differential_privacy_epsilon"] == 0.5
    assert config["privacy"]["retention_days"] == 30
