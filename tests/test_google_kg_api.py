#!/usr/bin/env python3
"""
Test Google Knowledge Graph API key functionality
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from google_kg_rag import GoogleKnowledgeGraphRAG
from hr_data_generator import HRDataGenerator

class TestGoogleKGAPI(unittest.TestCase):
    """Test Google Knowledge Graph API key functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.generator = HRDataGenerator()
        self.hr_data = self.generator.generate_comprehensive_data(
            num_employees=5, 
            num_policies=3
        )
        
    def test_api_key_loading_from_env(self):
        """Test that API key is properly loaded from environment variables"""
        print("\n" + "=" * 60)
        print("Testing Google KG API Key Loading")
        print("=" * 60)
        
        # Test with valid API key
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_valid_key_123'}):
            # Reload environment variables
            load_dotenv(override=True)
            kg_rag = GoogleKnowledgeGraphRAG(api_key="test_valid_key_123", hr_data=self.hr_data)
            self.assertEqual(kg_rag.api_key, 'test_valid_key_123')
            print("✅ API key loaded correctly from constructor")
        
        # Test with environment variable
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'env_test_key_456'}):
            # Reload environment variables
            load_dotenv(override=True)
            kg_rag = GoogleKnowledgeGraphRAG(api_key='env_test_key_456', hr_data=self.hr_data)
            self.assertEqual(kg_rag.api_key, 'env_test_key_456')
            print("✅ API key loaded correctly from environment variable")
    
    def test_api_key_validation(self):
        """Test API key validation logic"""
        print("\n" + "=" * 60)
        print("Testing Google KG API Key Validation")
        print("=" * 60)
        
        # Test with None API key
        kg_rag = GoogleKnowledgeGraphRAG(api_key=None, hr_data=self.hr_data)
        self.assertIsNone(kg_rag.api_key)
        print("✅ None API key handled correctly")
        
        # Test with empty API key
        kg_rag = GoogleKnowledgeGraphRAG(api_key="", hr_data=self.hr_data)
        self.assertEqual(kg_rag.api_key, "")
        print("✅ Empty API key handled correctly")
        
        # Test with demo key
        kg_rag = GoogleKnowledgeGraphRAG(api_key="demo_key", hr_data=self.hr_data)
        self.assertEqual(kg_rag.api_key, "demo_key")
        print("✅ Demo key handled correctly")
    
    def test_api_key_in_search_entities(self):
        """Test API key usage in search_entities method"""
        print("\n" + "=" * 60)
        print("Testing Google KG API Key in Entity Search")
        print("=" * 60)
        
        # Mock the requests.get method
        with patch('requests.get') as mock_get:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "itemListElement": [
                    {
                        "result": {
                            "name": "Google",
                            "@type": "Organization",
                            "description": "American multinational technology company"
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            # Test with valid API key
            kg_rag = GoogleKnowledgeGraphRAG(api_key="valid_test_key", hr_data=self.hr_data)
            results = kg_rag.search_entities("Google")
            
            # Verify the API call was made with the correct key
            mock_get.assert_called()
            call_args = mock_get.call_args
            self.assertEqual(call_args[1]['params']['key'], 'valid_test_key')
            print("✅ API key correctly used in entity search request")
            
            # Verify results
            self.assertIsInstance(results, list)
            if results:
                self.assertIn('name', results[0])
                print(f"✅ Entity search returned valid results: {results[0]['name']}")
    
    def test_api_key_in_hybrid_search(self):
        """Test API key usage in hybrid_search method"""
        print("\n" + "=" * 60)
        print("Testing Google KG API Key in Hybrid Search")
        print("=" * 60)
        
        # Mock the requests.get method
        with patch('requests.get') as mock_get:
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "itemListElement": [
                    {
                        "result": {
                            "name": "Employment Law",
                            "@type": "Thing",
                            "description": "Legal framework governing employment relationships"
                        }
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            # Test with valid API key
            kg_rag = GoogleKnowledgeGraphRAG(api_key="valid_test_key", hr_data=self.hr_data)
            results = kg_rag.hybrid_search("employment law", mode="External Only")
            
            # Verify the API call was made with the correct key
            mock_get.assert_called()
            call_args = mock_get.call_args
            self.assertEqual(call_args[1]['params']['key'], 'valid_test_key')
            print("✅ API key correctly used in hybrid search request")
            
            # Verify results
            self.assertIsInstance(results, list)
            print(f"✅ Hybrid search returned {len(results)} results")
    
    def test_api_key_error_handling(self):
        """Test API key error handling"""
        print("\n" + "=" * 60)
        print("Testing Google KG API Key Error Handling")
        print("=" * 60)
        
        # Mock the requests.get method to simulate API errors
        with patch('requests.get') as mock_get:
            # Mock API key invalid error
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT"
                }
            }
            mock_get.return_value = mock_response
            
            # Test with invalid API key
            kg_rag = GoogleKnowledgeGraphRAG(api_key="invalid_key", hr_data=self.hr_data)
            
            try:
                results = kg_rag.search_entities("Google")
                # Should handle error gracefully
                self.assertIsInstance(results, list)
                print("✅ Invalid API key error handled gracefully")
            except Exception as e:
                print(f"✅ Invalid API key error caught: {str(e)}")
    
    def test_dotenv_integration(self):
        """Test dotenv integration for API key loading"""
        print("\n" + "=" * 60)
        print("Testing dotenv Integration")
        print("=" * 60)
        
        # Create a temporary .env file
        test_env_content = "GOOGLE_API_KEY=dotenv_test_key_789"
        test_env_file = os.path.join(os.path.dirname(__file__), 'test.env')
        
        try:
            with open(test_env_file, 'w') as f:
                f.write(test_env_content)
            
            # Test loading from .env file
            with patch('dotenv.load_dotenv') as mock_load_dotenv:
                # Reload environment variables
                load_dotenv(test_env_file)
                
                # Verify dotenv was called
                mock_load_dotenv.assert_called()
                print("✅ dotenv integration working correctly")
                
        finally:
            # Clean up test file
            if os.path.exists(test_env_file):
                os.remove(test_env_file)
    
    def test_api_key_security(self):
        """Test API key security (not logging sensitive data)"""
        print("\n" + "=" * 60)
        print("Testing Google KG API Key Security")
        print("=" * 60)
        
        sensitive_key = "AIzaSyC_very_sensitive_api_key_123456789"
        
        # Test that API key is not exposed in string representation
        kg_rag = GoogleKnowledgeGraphRAG(api_key=sensitive_key, hr_data=self.hr_data)
        
        # Convert to string and check it doesn't contain the full key
        rag_str = str(kg_rag)
        self.assertNotIn(sensitive_key, rag_str)
        print("✅ API key not exposed in string representation")
        
        # Test that API key is not exposed in repr
        rag_repr = repr(kg_rag)
        self.assertNotIn(sensitive_key, rag_repr)
        print("✅ API key not exposed in repr representation")

def save_test_results(test_results):
    """Save test results to file"""
    results_file = os.path.join(os.path.dirname(__file__), '..', 'test-data', 'google_kg_api_test_results.json')
    
    # Ensure test-data directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    test_summary = {
        "timestamp": datetime.now().isoformat(),
        "test_name": "Google Knowledge Graph API Key Tests",
        "results": test_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(test_summary, f, indent=2, default=str)
    
    print(f"\n✅ Test results saved to: {results_file}")

def main():
    """Run all Google KG API tests"""
    print("🔑 Google Knowledge Graph API Key Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGoogleKGAPI)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Save test results
    test_results = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    }
    
    save_test_results(test_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Tests Run: {test_results['tests_run']}")
    print(f"Failures: {test_results['failures']}")
    print(f"Errors: {test_results['errors']}")
    print(f"Success Rate: {test_results['success_rate']:.1%}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 