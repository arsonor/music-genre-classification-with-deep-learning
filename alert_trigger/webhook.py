"""
Webhook service for triggering Prefect flows based on alerts.
"""
import os

import requests
from flask import Flask, request

app = Flask(__name__)

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
FLOW_NAME = os.getenv("PREFECT_FLOW_NAME", "retrain_model")

# Request timeout in seconds
REQUEST_TIMEOUT = 30


@app.route("/trigger", methods=["POST"])
def trigger():
    """
    Webhook endpoint to receive alerts and trigger retraining flow.

    Returns:
        tuple: Response message and HTTP status code
    """
    data = request.json
    print("Received alert:", data)

    # Check if this is an accuracy alert
    if data and "alerts" in data:
        for alert in data["alerts"]:
            if alert.get("labels", {}).get("alertname") == "AccuracyBelowThreshold":
                print("Triggering Prefect retrain flow...")
                trigger_prefect_flow()
                break

    return "OK", 200


def trigger_prefect_flow():
    """
    Trigger a Prefect flow deployment for model retraining.

    Makes an HTTP POST request to the Prefect API to create a new flow run.
    """
    # Call Prefect Deployment to trigger the flow
    # Replace with your deployment ID or use the name+version API
    deployment_id = os.getenv(
        "PREFECT_DEPLOYMENT_ID", "edf5202f-d8b6-4145-bc05-e75e15ff0417"
    )

    try:
        response = requests.post(
            f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run",
            json={"parameters": {}},
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 200:
            print("Prefect flow triggered successfully")
        else:
            print(
                f"Failed to trigger Prefect flow: {response.status_code} - {response.text}"
            )

    except requests.exceptions.Timeout:
        print(f"Request to Prefect API timed out after {REQUEST_TIMEOUT} seconds")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Prefect API: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5080, debug=False)
