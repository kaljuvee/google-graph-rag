#!/usr/bin/env python3
"""
Run all RAG implementation tests and generate comprehensive report
"""

import sys
import os
import json
import subprocess
from datetime import datetime
import traceback

def run_test_module(test_module):
    """Run a specific test module and capture results"""
    print(f"\n{'='*80}")
    print(f"RUNNING: {test_module}")
    print(f"{'='*80}")
    
    try:
        # Run the test module
        result = subprocess.run(
            [sys.executable, test_module],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return {
            'module': test_module,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"Test {test_module} timed out after 5 minutes")
        return {
            'module': test_module,
            'success': False,
            'error': 'Timeout after 5 minutes',
            'return_code': -1
        }
    except Exception as e:
        print(f"Error running {test_module}: {str(e)}")
        return {
            'module': test_module,
            'success': False,
            'error': str(e),
            'return_code': -1
        }

def load_test_results():
    """Load individual test result files"""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'test-results')
    
    test_result_files = [
        'vector_rag_test_results.json',
        'chroma_rag_test_results.json',
        'neo4j_rag_test_results.json'
    ]
    
    loaded_results = {}
    
    for result_file in test_result_files:
        file_path = os.path.join(results_dir, result_file)
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    loaded_results[result_file] = json.load(f)
                print(f"Loaded results from {result_file}")
            else:
                print(f"Result file not found: {result_file}")
        except Exception as e:
            print(f"Error loading {result_file}: {str(e)}")
    
    return loaded_results

def generate_comprehensive_report(test_execution_results, test_data_results):
    """Generate comprehensive test report"""
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE TEST REPORT")
    print(f"{'='*80}")
    
    # Create test results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'test-results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare comprehensive report
    report = {
        'report_timestamp': datetime.now().isoformat(),
        'test_execution_summary': {
            'total_test_modules': len(test_execution_results),
            'successful_modules': sum(1 for r in test_execution_results if r['success']),
            'failed_modules': sum(1 for r in test_execution_results if not r['success']),
            'execution_details': test_execution_results
        },
        'test_data_results': test_data_results,
        'overall_summary': {}
    }
    
    # Calculate overall statistics
    total_queries = 0
    successful_queries = 0
    
    for test_file, test_data in test_data_results.items():
        if 'summary' in test_data:
            summary = test_data['summary']
            total_queries += summary.get('total_queries', 0)
            successful_queries += summary.get('successful_queries', 0)
    
    report['overall_summary'] = {
        'total_test_modules': len(test_execution_results),
        'successful_test_modules': sum(1 for r in test_execution_results if r['success']),
        'module_success_rate': sum(1 for r in test_execution_results if r['success']) / len(test_execution_results) if test_execution_results else 0,
        'total_queries_across_all_tests': total_queries,
        'successful_queries_across_all_tests': successful_queries,
        'overall_query_success_rate': successful_queries / total_queries if total_queries > 0 else 0
    }
    
    # Save comprehensive report
    report_file = os.path.join(results_dir, 'comprehensive_test_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Comprehensive report saved to: {report_file}")
    
    # Generate human-readable summary
    generate_summary_report(report, results_dir)
    
    return report

def generate_summary_report(report, results_dir):
    """Generate human-readable summary report"""
    
    summary_file = os.path.join(results_dir, 'test_summary_report.txt')
    
    with open(summary_file, 'w') as f:
        f.write("GOOGLE GRAPH RAG MVP - TEST SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Report Generated: {report['report_timestamp']}\n\n")
        
        # Overall Summary
        overall = report['overall_summary']
        f.write("OVERALL TEST RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Modules Run: {overall['total_test_modules']}\n")
        f.write(f"Successful Modules: {overall['successful_test_modules']}\n")
        f.write(f"Module Success Rate: {overall['module_success_rate']:.1%}\n")
        f.write(f"Total Queries Tested: {overall['total_queries_across_all_tests']}\n")
        f.write(f"Successful Queries: {overall['successful_queries_across_all_tests']}\n")
        f.write(f"Overall Query Success Rate: {overall['overall_query_success_rate']:.1%}\n\n")
        
        # Individual Test Results
        f.write("INDIVIDUAL TEST MODULE RESULTS\n")
        f.write("-" * 40 + "\n")
        
        for test_result in report['test_execution_summary']['execution_details']:
            module_name = test_result['module'].replace('.py', '').replace('test_', '').replace('_', ' ').title()
            status = "PASSED" if test_result['success'] else "FAILED"
            f.write(f"{module_name}: {status}\n")
            
            if not test_result['success']:
                if 'error' in test_result:
                    f.write(f"  Error: {test_result['error']}\n")
                if 'return_code' in test_result:
                    f.write(f"  Return Code: {test_result['return_code']}\n")
        
        f.write("\n")
        
        # Detailed Results by RAG Type
        f.write("DETAILED RESULTS BY RAG TYPE\n")
        f.write("-" * 40 + "\n")
        
        for test_file, test_data in report['test_data_results'].items():
            if 'summary' in test_data:
                rag_type = test_file.replace('_test_results.json', '').replace('_', ' ').title()
                summary = test_data['summary']
                
                f.write(f"\n{rag_type}:\n")
                f.write(f"  Total Queries: {summary.get('total_queries', 0)}\n")
                f.write(f"  Successful Queries: {summary.get('successful_queries', 0)}\n")
                f.write(f"  Success Rate: {summary.get('successful_queries', 0) / max(1, summary.get('total_queries', 1)) * 100:.1f}%\n")
                
                if 'average_score' in summary:
                    f.write(f"  Average Score: {summary['average_score']:.3f}\n")
                if 'average_similarity' in summary:
                    f.write(f"  Average Similarity: {summary['average_similarity']:.3f}\n")
                if 'build_time' in summary:
                    f.write(f"  Build Time: {summary['build_time']:.2f}s\n")
                if 'avg_query_time' in summary:
                    f.write(f"  Avg Query Time: {summary['avg_query_time']:.3f}s\n")
        
        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("End of Report\n")
    
    print(f"Human-readable summary saved to: {summary_file}")

def install_dependencies():
    """Install required dependencies for testing"""
    print("Checking and installing dependencies...")
    
    required_packages = [
        'sentence-transformers',
        'faiss-cpu',
        'chromadb',
        'networkx',
        'plotly'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is available")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")

def main():
    """Run all tests and generate comprehensive report"""
    print("GOOGLE GRAPH RAG MVP - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test run started at: {datetime.now().isoformat()}")
    
    # Install dependencies
    install_dependencies()
    
    # Define test modules to run
    test_modules = [
        'test_vector_rag.py',
        'test_chroma_rag.py',
        'test_neo4j_rag.py'
    ]
    
    # Run all test modules
    test_execution_results = []
    
    for test_module in test_modules:
        result = run_test_module(test_module)
        test_execution_results.append(result)
    
    # Load individual test results
    print(f"\n{'='*80}")
    print("LOADING INDIVIDUAL TEST RESULTS")
    print(f"{'='*80}")
    
    test_data_results = load_test_results()
    
    # Generate comprehensive report
    comprehensive_report = generate_comprehensive_report(test_execution_results, test_data_results)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*80}")
    
    overall = comprehensive_report['overall_summary']
    print(f"Test Modules: {overall['successful_test_modules']}/{overall['total_test_modules']} passed")
    print(f"Module Success Rate: {overall['module_success_rate']:.1%}")
    print(f"Total Queries: {overall['total_queries_across_all_tests']}")
    print(f"Query Success Rate: {overall['overall_query_success_rate']:.1%}")
    
    # Determine overall test status
    if overall['module_success_rate'] >= 0.8 and overall['overall_query_success_rate'] >= 0.7:
        print("\n🎉 OVERALL TEST STATUS: PASSED")
        print("The Google Graph RAG MVP is ready for deployment!")
    elif overall['module_success_rate'] >= 0.6:
        print("\n⚠️  OVERALL TEST STATUS: PARTIAL SUCCESS")
        print("Some issues detected, but core functionality is working.")
    else:
        print("\n❌ OVERALL TEST STATUS: FAILED")
        print("Significant issues detected. Review test results before deployment.")
    
    print(f"\nTest run completed at: {datetime.now().isoformat()}")
    print("=" * 80)

if __name__ == "__main__":
    main()

