from flask import Flask, request
import requests
import os

app = Flask(__name__)

PREFECT_API_URL = os.getenv("PREFECT_API_URL", "http://prefect-server:4200/api")
FLOW_NAME = os.getenv("PREFECT_FLOW_NAME", "retrain_model")

@app.route("/trigger", methods=["POST"])
def trigger():
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
    # Call Prefect Deployment to trigger the flow
    # Replace with your deployment ID or use the name+version API
    deployment_id = os.getenv("PREFECT_DEPLOYMENT_ID", "YOUR_DEPLOYMENT_ID")

    response = requests.post(
        f"{PREFECT_API_URL}/deployments/{deployment_id}/create_flow_run",
        json={"parameters": {}}
    )

    if response.status_code == 200:
        print("Prefect flow triggered successfully")
    else:
        print("Failed to trigger Prefect flow:", response.text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5080)
