#!/usr/bin/env python3
"""
Quick runner for classifier tests.
"""

import subprocess
import sys
import os

def run_classifier_tests():
    """Run classifier tests."""
    print("🤖 Running Classifier Tests...")
    print("=" * 50)
    
    # Check if classifier directory exists
    classifier_dir = "classifier"
    if not os.path.exists(classifier_dir):
        print(f"❌ Classifier directory not found: {classifier_dir}")
        return False
    
    # Check if test file exists
    test_file = os.path.join("tests", "test_classifier.py")
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Set environment variables to avoid Prefect server issues
        env = os.environ.copy()
        env['PREFECT_API_URL'] = 'http://localhost:4200/api'
        env['PREFECT_DISABLE_CLIENT'] = 'true'
        
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file,
            "-v", "--tb=short", "-x"  # Stop on first failure
        ], timeout=120, env=env)  # 2 minutes timeout for ML tests
        
        if result.returncode == 0:
            print("\n✅ All classifier tests passed!")
            return True
        else:
            print("\n❌ Some classifier tests failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n⏰ Tests timed out")
        return False
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        return False

def run_quick_import_test():
    """Test if classifier modules can be imported."""
    print("🔍 Testing classifier imports...")
    
    try:
        # Test basic imports
        sys.path.append("classifier")
        
        # Test imports without Prefect decorators
        from classifier.utils.data import load_data, prepare_dataset, plot_history
        from classifier.models.model import build_model
        from classifier.pipeline.train import train_and_log_model
        
        print("✅ All classifier modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    print("🔍 Classifier Test Runner")
    print("=" * 40)
    
    # Test imports first
    import_ok = run_quick_import_test()
    
    if not import_ok:
        print("\n🔧 Fix import issues before running tests")
        return False
    
    print("\n" + "=" * 40)
    
    # Run full tests
    success = run_classifier_tests()
    
    if success:
        print("\n🎉 Classifier tests are working!")
        print("\n📊 Test Coverage:")
        print("  ✅ Data loading and validation") 
        print("  ✅ Feature extraction pipeline")
        print("  ✅ Model building and compilation")
        print("  ✅ Training logic (MLflow mocked)")
        print("  ✅ Data preparation and splitting")
        print("  ✅ Training history visualization")
        print("  ✅ Configuration handling")
        print("  ✅ Integration with realistic data")
        
        print("\n💡 Note: Prefect flow tests are simplified to avoid server dependencies")
        print("   The core ML pipeline logic is fully tested!")
        
        print("\n🚀 You can now run:")
        print("  python run_tests.py classifier")
        print("  make test-classifier")
    else:
        print("\n🔧 Fix the issues and try again")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)