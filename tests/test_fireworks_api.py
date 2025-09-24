import json
import os

import requests

from eval_protocol.auth import get_fireworks_account_id, get_fireworks_api_key


def test_fireworks_api():
    # Get API key using the new auth module
    api_key = get_fireworks_api_key()
    if api_key:
        print(f"API key retrieved via auth module: {api_key[:4]}...{api_key[-4:]}")
    else:
        print("No API key retrieved via auth module.")

    # Get account ID using the new auth module
    account_id = get_fireworks_account_id()
    if account_id:
        print(f"Account ID retrieved via auth module: {account_id}")
    else:
        print("No account ID retrieved via auth module.")

    # Ensure api_key is not None for header construction, default to empty string if None
    effective_api_key = api_key if api_key is not None else ""

    # Test API connection
    try:
        # Try listing models to verify API connectivity
        headers = {"Authorization": f"Bearer {effective_api_key}"}
        base_url = "https://api.fireworks.ai/v1"

        # Check if models endpoint works (to verify API connection)
        models_url = f"{base_url}/models?limit=1"
        print(f"Testing models endpoint: {models_url}")
        response = requests.get(models_url, headers=headers)
        print(f"Response: {response.status_code} - {response.reason}")
        if response.status_code == 200:
            print("Successfully connected to Fireworks API")
        else:
            print(f"Error response: {response.text}")

        if account_id:
            # Check if the evaluations endpoint is available
            eval_url = f"{base_url}/accounts/{account_id}/evaluations"
            print(f"Testing evaluations endpoint: {eval_url}")
            response = requests.get(eval_url, headers=headers)
            print(f"Response: {response.status_code} - {response.reason}")
            if response.status_code != 200:
                print(f"Error response: {response.text}")

            # Check if there's an evaluators endpoint
            evaluators_url = f"{base_url}/accounts/{account_id}/evaluators"
            print(f"Testing evaluators endpoint: {evaluators_url}")
            response = requests.get(evaluators_url, headers=headers)
            print(f"Response: {response.status_code} - {response.reason}")
            if response.status_code != 200:
                print(f"Error response: {response.text}")

            # Look for alternate endpoints
            for endpoint in ["evaluation", "evaluator"]:
                url = f"{base_url}/accounts/{account_id}/{endpoint}"
                print(f"Testing alternate endpoint: {url}")
                response = requests.get(url, headers=headers)
                print(f"Response: {response.status_code} - {response.reason}")

    except Exception as e:
        print(f"Error connecting to Fireworks API: {str(e)}")
