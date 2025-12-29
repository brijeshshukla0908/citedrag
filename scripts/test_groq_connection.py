# test_groq_connection.py
"""
Test Groq API Connection
========================
Run this script to verify your Groq API key is working correctly.

Usage:
    python scripts/test_groq_connection.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

def test_groq_connection():
    """Test Groq API connection and display key info"""
    
    print("\n" + "="*60)
    print("CitedRAG - Groq API Connection Test")
    print("="*60 + "\n")
    
    # Check if API key exists
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found in .env file")
        print("\nSteps to fix:")
        print("1. Copy .env.example to .env")
        print("2. Get API key from https://console.groq.com")
        print("3. Add to .env: GROQ_API_KEY=your_key_here")
        return False
    
    # Mask API key for display
    masked_key = api_key[:8] + "..." + api_key[-4:]
    print(f"✅ API Key found: {masked_key}")
    
    # Test API connection
    try:
        print("\n🔄 Testing API connection...")
        client = Groq(api_key=api_key)
        
        # Send test request
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello from CitedRAG!' if you can read this."}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        # Display response
        message = response.choices[0].message.content
        print(f"\n✅ API Connection Successful!")
        print(f"\n📝 Response: {message}")
        
        # Display usage stats
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"\n📊 Token Usage:")
            print(f"   - Prompt tokens: {usage.prompt_tokens}")
            print(f"   - Completion tokens: {usage.completion_tokens}")
            print(f"   - Total tokens: {usage.total_tokens}")
        
        print("\n" + "="*60)
        print("✅ All tests passed! You're ready to build CitedRAG.")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: API connection failed")
        print(f"\nError details: {str(e)}")
        print("\nPossible issues:")
        print("- Invalid API key")
        print("- No internet connection")
        print("- Groq service temporarily down")
        print("\nVisit https://console.groq.com to verify your key")
        return False

if __name__ == "__main__":
    success = test_groq_connection()
    sys.exit(0 if success else 1)
