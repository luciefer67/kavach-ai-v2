import requests
import time
import json

LOKI_URL = "http://localhost:3100/loki/api/v1/push"

def send_to_loki(log):
    payload = {
        "streams": [{
            "stream": {
                "job": "kavach-ai",
                "severity": log["severity"]
            },
            "values": [
                [str(int(time.time() * 1e9)), json.dumps(log)]
            ]
        }]
    }
    requests.post(LOKI_URL, json=payload)
