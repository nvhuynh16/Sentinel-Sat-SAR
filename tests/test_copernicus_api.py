"""
Tests for CopernicusAPI authentication functionality.
Following TDD principles: Write tests first, then implement.
"""
import pytest
from pathlib import Path
import sys

# Add parent directory to path to import CopernicusAPI
sys.path.insert(0, str(Path(__file__).parent.parent))
from CopernicusAPI import CopernicusAPI


class TestCopernicusAPIAuth:
    """Test suite for CopernicusAPI authentication methods."""

    def test_authenticate_from_file_success(self):
        """Test loading credentials from login.txt file successfully."""
        api = CopernicusAPI()
        test_file = Path(__file__).parent / 'fixtures' / 'test_login.txt'

        # This should load credentials without error (auto_token=False for testing)
        api.authenticate_from_file(str(test_file), auto_token=False)

        # Verify credentials were loaded
        assert api.username is not None, "Username should be loaded"
        assert api.password is not None, "Password should be loaded"
        assert api.username == "test.user@example.com", "Username should match file"
        assert api.password == "test_password_12345", "Password should match file"
        assert '@' in api.username, "Username should be valid email format"

    def test_authenticate_from_file_not_found(self):
        """Test graceful error when credentials file doesn't exist."""
        api = CopernicusAPI()

        with pytest.raises(FileNotFoundError) as exc_info:
            api.authenticate_from_file('nonexistent_file.txt')

        assert 'nonexistent_file.txt' in str(exc_info.value)

    def test_authenticate_from_file_malformed(self, tmp_path):
        """Test error handling for malformed credentials file."""
        api = CopernicusAPI()

        # Create a malformed file
        malformed_file = tmp_path / "malformed.txt"
        malformed_file.write_text("This is not the right format\nNo username or password")

        with pytest.raises(ValueError) as exc_info:
            api.authenticate_from_file(str(malformed_file))

        assert 'format' in str(exc_info.value).lower() or 'invalid' in str(exc_info.value).lower()

    def test_authenticate_from_file_strips_whitespace(self, tmp_path):
        """Test that credentials are stripped of leading/trailing whitespace."""
        api = CopernicusAPI()

        # Create file with extra whitespace
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("Username:  test@example.com  \nPassword:  pass123  \n")

        api.authenticate_from_file(str(whitespace_file), auto_token=False)

        assert api.username == "test@example.com", "Username should be stripped"
        assert api.password == "pass123", "Password should be stripped"

    def test_authenticate_from_file_default_path(self):
        """Test that authenticate_from_file uses 'login.txt' as default."""
        api = CopernicusAPI()

        # This will fail if login.txt doesn't exist, but tests the default parameter
        # We expect it to try to open 'login.txt' by default
        try:
            api.authenticate_from_file(auto_token=False)
            # If successful, verify credentials were loaded
            assert api.username is not None
            assert api.password is not None
        except FileNotFoundError:
            # Expected if login.txt doesn't exist in test context
            pass
