import requests
import json

def test_math():
    url = "http://127.0.0.1:8000/api/chat"
    # A specific calculation unlikely to be pre-canned
    payload = {"message": "Calculate 25 * 25"}
    
    print(f"Sending: {payload['message']}")
    try:
        response = requests.post(url, json=payload, timeout=30)
        data = response.json()
        print(f"DEBUG RAW RESPONSE: {json.dumps(data, indent=2)}")
        print(f"Response: {data['response']}")
        print("Thought Process:")
        for t in data['thought_process']:
            print(f" - [{t['action']}] {t['thought']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_math()
