#!/usr/bin/env python3
"""
Test runner script for music genre classification API tests.
This script helps set up the environment and run tests with proper configuration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def setup_python_path():
    """Add the api directory to Python path."""
    project_root = Path(__file__).parent
    api_path = project_root / "api"
    if str(api_path) not in sys.path:
        sys.path.insert(0, str(api_path))


def install_dependencies():
    """Install required dependencies for testing."""
    print("Installing dependencies...")
    
    # Install main dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "api/requirements.txt"
    ], check=True)
    
    # Install monitoring dependencies
    if os.path.exists("monitoring/requirements.txt"):
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "monitoring/requirements.txt"
        ], check=True)
    
    # Install test dependencies
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "tests/requirements-test.txt"
    ], check=True)
    
    print("Dependencies installed successfully!")


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests based on specified type."""
    setup_python_path()
    
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test path
    if test_type == "all":
        cmd.append("tests/")
    elif test_type == "unit":
        cmd.extend(["tests/", "-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["tests/", "-m", "integration"])
    elif test_type == "service":
        cmd.append("tests/test_genre_prediction_service.py")
    elif test_type == "server":
        cmd.append("tests/test_server.py")
    elif test_type == "monitoring":
        cmd.append("tests/test_monitoring.py")
    else:
        cmd.append(f"tests/test_{test_type}.py")
    
    # Add verbosity
    if verbose:
        cmd.extend(["-v", "-s"])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=api",
            "--cov=monitoring",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False


def clean_artifacts():
    """Clean up test artifacts and cache files."""
    print("Cleaning up test artifacts...")
    
    artifacts_to_remove = [
        "htmlcov/",
        ".coverage",
        ".pytest_cache/",
        "__pycache__/",
        "tests/__pycache__/",
        "api/__pycache__/"
    ]
    
    for artifact in artifacts_to_remove:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                subprocess.run(["rm", "-rf", artifact])
            else:
                os.remove(artifact)
    
    # Remove .pyc files
    subprocess.run([
        "find", ".", "-name", "*.pyc", "-delete"
    ], stderr=subprocess.DEVNULL)
    
    print("Cleanup completed!")


def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run tests for music genre classification API")
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "service", "server", "monitoring"],
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Run tests with coverage report"
    )
    
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install dependencies before running tests"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean up test artifacts and cache files"
    )
    
    args = parser.parse_args()
    
    if args.clean:
        clean_artifacts()
        return
    
    if args.install:
        install_dependencies()
    
    print(f"Running {args.test_type} tests...")
    success = run_tests(args.test_type, args.verbose, args.coverage)
    
    if success:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()