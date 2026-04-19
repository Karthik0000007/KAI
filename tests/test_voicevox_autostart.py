"""
Unit tests for VOICEVOX auto-start detection and launch.

Tests Requirements 7.4 and 7.5:
- 7.4: WHERE VOICEVOX is not running, THE TTS_Engine SHALL attempt to start it automatically
- 7.5: THE TTS_Engine SHALL detect VOICEVOX installation in common paths on Windows, macOS, and Linux
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVoicevoxAutostart(unittest.TestCase):
    """Test VOICEVOX auto-start detection and launch functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import the function we're testing with mocked dependencies
        self.mock_modules = {
            'TTS': MagicMock(),
            'TTS.api': MagicMock(),
            'TTS.utils.radam': MagicMock(),
            'torch': MagicMock(),
            'langdetect': MagicMock(),
            'fugashi': MagicMock(),
        }
        
        # Patch sys.modules to avoid importing heavy dependencies
        self.patcher = patch.dict('sys.modules', self.mock_modules)
        self.patcher.start()
        
        # Now we can import the module
        import core.tts as tts_module
        self.tts = tts_module
        self.tts._voicevox_process = None
    
    def tearDown(self):
        """Clean up after each test."""
        self.patcher.stop()
        if hasattr(self, 'tts'):
            self.tts._voicevox_process = None
    
    @patch('requests.get')
    def test_voicevox_already_running(self, mock_get):
        """Test that function returns True when VOICEVOX is already running."""
        # Mock successful version check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertTrue(result)
        mock_get.assert_called_once()
    
    @patch('subprocess.Popen')
    @patch('requests.get')
    @patch('shutil.which')
    @patch('time.sleep')
    def test_voicevox_found_in_path(self, mock_sleep, mock_which, mock_get, mock_popen):
        """Test that VOICEVOX is found and started from PATH."""
        # Mock VOICEVOX not running initially
        mock_get.side_effect = [
            Exception("Not running"),  # Initial check fails
            MagicMock(status_code=200, text="0.14.0")  # After start succeeds
        ]
        
        # Mock finding VOICEVOX in PATH
        mock_which.return_value = "/usr/local/bin/voicevox"
        
        # Mock subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertTrue(result)
        mock_which.assert_called()
        mock_popen.assert_called_once()
    
    @patch('subprocess.Popen')
    @patch('requests.get')
    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('os.name', 'nt')
    @patch('time.sleep')
    def test_voicevox_found_in_windows_common_location(self, mock_sleep, mock_isfile, mock_which, mock_get, mock_popen):
        """Test that VOICEVOX is found in Windows common install location."""
        # Mock VOICEVOX not running initially
        mock_get.side_effect = [
            Exception("Not running"),  # Initial check fails
            MagicMock(status_code=200, text="0.14.0")  # After start succeeds
        ]
        
        # Mock not finding in PATH
        mock_which.return_value = None
        
        # Mock finding in common Windows location
        def isfile_side_effect(path):
            return "VOICEVOX\\run.exe" in str(path)
        mock_isfile.side_effect = isfile_side_effect
        
        # Mock subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertTrue(result)
        mock_popen.assert_called_once()
    
    @patch('subprocess.Popen')
    @patch('requests.get')
    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('os.name', 'posix')
    @patch('platform.system')
    @patch('time.sleep')
    def test_voicevox_found_in_macos_common_location(self, mock_sleep, mock_system, mock_isfile, mock_which, mock_get, mock_popen):
        """Test that VOICEVOX is found in macOS common install location."""
        # Mock macOS
        mock_system.return_value = "Darwin"
        
        # Mock VOICEVOX not running initially
        mock_get.side_effect = [
            Exception("Not running"),  # Initial check fails
            MagicMock(status_code=200, text="0.14.0")  # After start succeeds
        ]
        
        # Mock not finding in PATH
        mock_which.return_value = None
        
        # Mock finding in common macOS location
        def isfile_side_effect(path):
            return "VOICEVOX.app/Contents/MacOS/run" in str(path)
        mock_isfile.side_effect = isfile_side_effect
        
        # Mock subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertTrue(result)
        mock_popen.assert_called_once()
    
    @patch('subprocess.Popen')
    @patch('requests.get')
    @patch('shutil.which')
    @patch('os.path.isfile')
    @patch('os.name', 'posix')
    @patch('platform.system')
    @patch('time.sleep')
    def test_voicevox_found_in_linux_common_location(self, mock_sleep, mock_system, mock_isfile, mock_which, mock_get, mock_popen):
        """Test that VOICEVOX is found in Linux common install location."""
        # Mock Linux
        mock_system.return_value = "Linux"
        
        # Mock VOICEVOX not running initially
        mock_get.side_effect = [
            Exception("Not running"),  # Initial check fails
            MagicMock(status_code=200, text="0.14.0")  # After start succeeds
        ]
        
        # Mock not finding in PATH
        mock_which.return_value = None
        
        # Mock finding in common Linux location
        def isfile_side_effect(path):
            return str(path) == "/opt/voicevox/run"
        mock_isfile.side_effect = isfile_side_effect
        
        # Mock subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertTrue(result)
        mock_popen.assert_called_once()
    
    @patch('requests.get')
    @patch('shutil.which')
    @patch('os.path.isfile')
    def test_voicevox_not_found(self, mock_isfile, mock_which, mock_get):
        """Test that function returns False when VOICEVOX is not found."""
        # Mock VOICEVOX not running
        mock_get.side_effect = Exception("Not running")
        
        # Mock not finding in PATH
        mock_which.return_value = None
        
        # Mock not finding in common locations
        mock_isfile.return_value = False
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertFalse(result)
    
    @patch('subprocess.Popen')
    @patch('requests.get')
    @patch('shutil.which')
    @patch('time.sleep')
    def test_voicevox_start_timeout(self, mock_sleep, mock_which, mock_get, mock_popen):
        """Test that function returns False when VOICEVOX doesn't respond in time."""
        # Mock VOICEVOX not running and never responding
        mock_get.side_effect = Exception("Not running")
        
        # Mock finding VOICEVOX in PATH
        mock_which.return_value = "/usr/local/bin/voicevox"
        
        # Mock subprocess
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertFalse(result)
        mock_popen.assert_called_once()
    
    @patch('subprocess.Popen')
    @patch('requests.get')
    @patch('shutil.which')
    def test_voicevox_start_permission_error(self, mock_which, mock_get, mock_popen):
        """Test that function handles permission errors gracefully."""
        # Mock VOICEVOX not running
        mock_get.side_effect = Exception("Not running")
        
        # Mock finding VOICEVOX in PATH
        mock_which.return_value = "/usr/local/bin/voicevox"
        
        # Mock permission error
        mock_popen.side_effect = PermissionError("Permission denied")
        
        result = self.tts._ensure_voicevox_running()
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
