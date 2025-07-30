#!/usr/bin/env python3
"""
Demo Runner Script
Executes all demo scripts with options for individual or comprehensive runs
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_demo(script_name, description):
    """Run a demo script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 Running {description}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=False, 
                              text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with return code: {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run RAG demo scripts")
    parser.add_argument("--vector", action="store_true", 
                       help="Run basic vector search demo")
    parser.add_argument("--graph", action="store_true", 
                       help="Run graph-based search demo")
    parser.add_argument("--kg", action="store_true", 
                       help="Run Google Knowledge Graph demo")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive demo (all approaches)")
    parser.add_argument("--all", action="store_true", 
                       help="Run all individual demos")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.vector, args.graph, args.kg, args.comprehensive, args.all]):
        parser.print_help()
        print("\n💡 Examples:")
        print("  python run_all_demos.py --vector")
        print("  python run_all_demos.py --all")
        print("  python run_all_demos.py --comprehensive")
        return
    
    print("🎯 RAG Demo Runner")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("basic_vector_demo.py").exists():
        print("❌ Please run this script from the playground directory")
        return
    
    # Create test-data directory
    test_data_dir = Path("../test-data")
    test_data_dir.mkdir(exist_ok=True)
    print(f"📁 Test data will be saved to: {test_data_dir.absolute()}")
    
    success_count = 0
    total_count = 0
    
    # Run individual demos
    if args.vector or args.all:
        total_count += 1
        if run_demo("basic_vector_demo.py", "Basic Vector Search Demo"):
            success_count += 1
    
    if args.graph or args.all:
        total_count += 1
        if run_demo("graph_search_demo.py", "Graph-based Search Demo"):
            success_count += 1
    
    if args.kg or args.all:
        total_count += 1
        if run_demo("google_kg_demo.py", "Google Knowledge Graph Demo"):
            success_count += 1
    
    # Run comprehensive demo
    if args.comprehensive:
        total_count += 1
        if run_demo("comprehensive_demo.py", "Comprehensive RAG Demo"):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Demo Execution Summary")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {total_count - success_count}")
    print(f"📈 Success Rate: {(success_count/total_count)*100:.1f}%" if total_count > 0 else "N/A")
    
    if success_count > 0:
        print(f"\n📁 Check the test-data directory for results:")
        print(f"   {test_data_dir.absolute()}")
    
    print(f"\n🎉 Demo runner completed!")

if __name__ == "__main__":
    main() 