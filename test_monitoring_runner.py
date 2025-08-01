#!/usr/bin/env python3
"""
Quick runner for monitoring tests.
"""

import subprocess
import sys
import os

def run_monitoring_tests():
    """Run monitoring tests."""
    print("🧪 Running Monitoring Tests...")
    print("=" * 50)
    
    # Check if monitoring script exists
    monitoring_script = os.path.join("monitoring", "run_monitoring.py")
    if not os.path.exists(monitoring_script):
        print(f"❌ Monitoring script not found: {monitoring_script}")
        return False
    
    # Check if test file exists
    test_file = os.path.join("tests", "test_monitoring.py")
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file,
            "-v", "--tb=short"
        ], timeout=60)
        
        if result.returncode == 0:
            print("\n✅ All monitoring tests passed!")
            return True
        else:
            print("\n❌ Some monitoring tests failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Tests timed out")
        return False
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        return False

def main():
    """Main function."""
    print("🔍 Monitoring Test Runner")
    print("=" * 40)
    
    success = run_monitoring_tests()
    
    if success:
        print("\n🎉 Monitoring tests are working!")
        print("\n📊 Test Coverage:")
        print("  ✅ Prometheus metric formatting")
        print("  ✅ Flask endpoint responses") 
        print("  ✅ Data processing logic")
        print("  ✅ Error handling")
        print("  ✅ Metric name sanitization")
        
        print("\n🚀 You can now run:")
        print("  python run_tests.py monitoring")
        print("  make test-monitoring")
    else:
        print("\n🔧 Fix the issues and try again")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)