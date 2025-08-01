#!/bin/bash

# Quick setup script for music genre classification tests
set -e

echo "🎵 Setting up Music Genre Classification Tests 🎵"
echo "=================================================="

# Create tests directory if it doesn't exist
if [ ! -d "tests" ]; then
    echo "📁 Creating tests directory..."
    mkdir -p tests
fi

# Create monitoring directory if it doesn't exist
if [ ! -d "monitoring" ]; then
    echo "📁 Creating monitoring directory..."
    mkdir -p monitoring
fi

# Check if we're in the right directory
if [ ! -d "api" ]; then
    echo "❌ Error: api/ directory not found. Please run this script from the project root."
    exit 1
fi

# Install main dependencies
echo "📦 Installing main API dependencies..."
pip install -r api/requirements.txt

# Install monitoring dependencies if they exist
if [ -f "monitoring/requirements.txt" ]; then
    echo "📊 Installing monitoring dependencies..."
    pip install -r monitoring/requirements.txt
else
    echo "⚠️  monitoring/requirements.txt not found, skipping monitoring dependencies"
fi

# Install test dependencies
echo "🧪 Installing test dependencies..."
pip install -r tests/requirements-test.txt

# Make run_tests.py executable
if [ -f "run_tests.py" ]; then
    chmod +x run_tests.py
    echo "✅ Made run_tests.py executable"
fi

# Make test runner scripts executable
if [ -f "test_monitoring_runner.py" ]; then
    chmod +x test_monitoring_runner.py
    echo "✅ Made test_monitoring_runner.py executable"
fi

# Run a quick test to verify setup
echo "🔍 Running quick test to verify setup..."
python -m pytest tests/ --collect-only -q

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup completed successfully!"
    echo ""
    echo "🚀 Quick start commands:"
    echo "  Run all tests:           python run_tests.py"
    echo "  Run with coverage:       python run_tests.py -c"
    echo "  Run verbose:             python run_tests.py -v"
    echo "  Run API tests:           python run_tests.py service"
    echo "  Run server tests:        python run_tests.py server"
    echo "  Run monitoring tests:    python run_tests.py monitoring"
    echo "  Clean artifacts:         python run_tests.py --clean"
    echo ""
    echo "📖 For more options, see: python run_tests.py --help"
    echo ""
    echo "🧪 Test categories available:"
    echo "  • API/Service tests (genre prediction logic)"
    echo "  • Server tests (Flask endpoint testing)"
    echo "  • Monitoring tests (Prometheus metrics)"
    echo "  • Integration tests (end-to-end workflows)"
else
    echo "❌ Setup verification failed. Please check the error messages above."
    exit 1
fi