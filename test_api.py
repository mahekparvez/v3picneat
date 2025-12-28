# test_api.py - Test your API before connecting Flutter

import requests
import sys

print("Testing Nutrition Tracker API...")
print("=" * 50)

# Test 1: Check if server is running
print("\n1. Checking if server is running...")
try:
    response = requests.get('https://pay-film-uncle-carl.trycloudflare.com')
    if response.status_code == 200:
        print("   ✓ Server is running!")
        print(f"   Response: {response.text}")
    else:
        print("   ✗ Server returned error")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Cannot connect to server: {e}")
    print("   Make sure you ran: python api.py")
    sys.exit(1)

# Test 2: Analyze an image
print("\n2. Testing image analysis...")
image_path = input("Enter path to test image (pizza/pasta/pancake): ")

try:
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post('https://pay-film-uncle-carl.trycloudflare.com', files=files)
    
    if response.status_code == 200:
        result = response.json()
        print("   ✓ Analysis successful!")
        print("\n   Results:")
        print("   " + "-" * 40)
        
        item = result['items'][0]
        print(f"   Food: {item['food']}")
        print(f"   Confidence: {item['conf']:.0%}")
        print(f"   Weight: {item['weight']}g")
        print(f"   Calories: {item['cal']} kcal")
        print(f"   Protein: {item['prot']}g")
        print(f"   Carbs: {item['carb']}g")
        print(f"   Fats: {item['fat']}g")
        print("   " + "-" * 40)
        print("\n   ✅ YOUR BACKEND IS WORKING!")
    else:
        print(f"   ✗ Error: {response.status_code}")
        print(f"   {response.text}")
except FileNotFoundError:
    print(f"   ✗ Image not found: {image_path}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 50)
print("Testing complete!")
