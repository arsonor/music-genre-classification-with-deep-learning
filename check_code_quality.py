#!/usr/bin/env python3
"""
Code quality checker script using black, isort, and pylint.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
            
    except Exception as e:
        print(f"💥 Error running {description}: {e}")
        return False


def get_python_files(directories):
    """Get all Python files in the specified directories."""
    python_files = []
    for directory in directories:
        if os.path.exists(directory):
            path = Path(directory)
            python_files.extend(path.rglob("*.py"))
    return [str(f) for f in python_files]


def check_isort(files, fix=False):
    """Check import sorting with isort."""
    if not files:
        print("⚠️  No Python files found for isort check")
        return True
    
    cmd = [sys.executable, "-m", "isort"]
    if not fix:
        cmd.append("--check-only")
        cmd.append("--diff")
    cmd.extend(files)
    
    description = "Import sorting (isort)" + (" - FIXING" if fix else " - CHECK")
    return run_command(cmd, description)


def check_black(files, fix=False):
    """Check code formatting with black."""
    if not files:
        print("⚠️  No Python files found for black check")
        return True
    
    cmd = [sys.executable, "-m", "black"]
    if not fix:
        cmd.append("--check")
        cmd.append("--diff")
    cmd.extend(files)
    
    description = "Code formatting (black)" + (" - FIXING" if fix else " - CHECK")
    return run_command(cmd, description)


def check_pylint(files):
    """Check code quality with pylint."""
    if not files:
        print("⚠️  No Python files found for pylint check")
        return True
    
    # Split files into smaller chunks to avoid command line length issues
    chunk_size = 10
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    
    all_passed = True
    for i, chunk in enumerate(file_chunks):
        cmd = [sys.executable, "-m", "pylint"] + chunk
        description = f"Code quality (pylint) - Chunk {i+1}/{len(file_chunks)}"
        
        if not run_command(cmd, description):
            all_passed = False
    
    return all_passed


def main():
    """Main function to run code quality checks."""
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument(
        "--fix", 
        action="store_true", 
        help="Fix issues with black and isort (pylint is check-only)"
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["alert_trigger", "api", "classifier", "monitoring", "tests"],
        help="Directories to check (default: api classifier monitoring tests)"
    )
    parser.add_argument(
        "--skip-pylint",
        action="store_true",
        help="Skip pylint checks (useful for quick formatting fixes)"
    )
    
    args = parser.parse_args()
    
    print("🔍 Code Quality Checker")
    print("=" * 40)
    print(f"Directories: {', '.join(args.directories)}")
    print(f"Mode: {'FIX' if args.fix else 'CHECK'}")
    
    # Get all Python files
    python_files = get_python_files(args.directories)
    print(f"Found {len(python_files)} Python files")
    
    if not python_files:
        print("❌ No Python files found in specified directories")
        return 1
    
    # Show first few files as sample
    print("Sample files:")
    for file in python_files[:5]:
        print(f"  - {file}")
    if len(python_files) > 5:
        print(f"  ... and {len(python_files) - 5} more")
    
    results = []
    
    # Run isort
    results.append(("isort", check_isort(python_files, fix=args.fix)))
    
    # Run black
    results.append(("black", check_black(python_files, fix=args.fix)))
    
    # Run pylint (check only)
    if not args.skip_pylint:
        results.append(("pylint", check_pylint(python_files)))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for tool, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{tool:10} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tools passed")
    
    if passed == total:
        print("🎉 All code quality checks passed!")
        return 0
    else:
        print("🔧 Some code quality issues found")
        if not args.fix:
            print("💡 Run with --fix to automatically fix black and isort issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())