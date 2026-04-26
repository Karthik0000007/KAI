"""
Pytest configuration and Hypothesis setup for Aegis test suite.

This module configures Hypothesis profiles for property-based testing:
- ci: 100 examples, verbose output (for CI/CD pipelines)
- dev: 20 examples, normal output (for local development)
- thorough: 1000 examples, verbose output (for comprehensive testing)

Usage:
    # Run with default profile (ci)
    pytest tests/

    # Run with specific profile
    pytest tests/ --hypothesis-profile=dev
    pytest tests/ --hypothesis-profile=thorough

    # Run only property-based tests
    pytest tests/ -m property
"""

from hypothesis import settings, Verbosity, HealthCheck

# ─── Hypothesis Profile Configuration ───────────────────────────────────────

# CI Profile: Balanced testing for continuous integration
# - 100 examples per property test
# - Verbose output for debugging failures
# - Suitable for automated testing pipelines
settings.register_profile(
    "ci",
    max_examples=100,
    verbosity=Verbosity.verbose,
    deadline=None,  # Disable deadline for slow operations (LLM, TTS)
    suppress_health_check=[HealthCheck.too_slow],  # Allow slow tests
)

# Dev Profile: Fast feedback for local development
# - 20 examples per property test
# - Normal output for quick iteration
# - Suitable for rapid development cycles
settings.register_profile(
    "dev",
    max_examples=20,
    verbosity=Verbosity.normal,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Thorough Profile: Comprehensive testing for release validation
# - 1000 examples per property test
# - Verbose output for detailed analysis
# - Suitable for pre-release validation and bug hunting
settings.register_profile(
    "thorough",
    max_examples=1000,
    verbosity=Verbosity.verbose,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load CI profile by default
# This can be overridden with --hypothesis-profile=<profile> flag
settings.load_profile("ci")
