#!/usr/bin/env python3
"""
Test Enhanced Realized Variance Implementation
==============================================
Quick test script for the fixed Enhanced RV computation
"""

from importlib import import_module

def run_test():
    """Run test with 3-symbol subset"""
    print("🧪 RUNNING ENHANCED RV TEST MODE")
    print("="*50)
    
    # Import the test function from main module
    yz_module = import_module('2_compute_enhanced_realized_variance')
    
    # Run the test
    success = yz_module.test_fixed_implementation()
    
    if success:
        print("\n✅ TEST PASSED - Implementation is working correctly!")
        print("💡 Ready to run full production processing")
    else:
        print("\n❌ TEST FAILED - Check implementation")
    
    return success

if __name__ == "__main__":
    run_test()