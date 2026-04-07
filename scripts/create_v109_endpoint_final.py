import os
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
IMAGE = "byoungj/sd:v109"
TEMPLATE_ID = "amfibofavv" # Use this as source

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def create_template():
    # Fetch source template first
    resp = requests.get(f"https://rest.runpod.io/v1/templates/{TEMPLATE_ID}", headers=headers)
    source = resp.json()
    
    query = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
      saveTemplate(input: $input) {
        id
        name
      }
    }
    """
    
    env = [{"key": k, "value": v} for k, v in source["env"].items()]
    # Add HF_TOKEN if missing
    if not any(e["key"] == "HF_TOKEN" for e in env):
        print(f"Adding HF_TOKEN to template env")
        env.append({"key": "HF_TOKEN", "value": HF_TOKEN})
    
    payload = {
        "name": f"sd-v109-{int(time.time())}",
        "imageName": IMAGE,
        "isServerless": True,
        "containerDiskInGb": 30,
        "env": env,
        "containerRegistryAuthId": source.get("containerRegistryAuthId"),
        "dockerArgs": "",
        "volumeInGb": 0
    }
    
    resp = requests.post(
        "https://api.runpod.io/graphql",
        headers={"content-type": "application/json"},
        params={"api_key": RUNPOD_API_KEY},
        json={"query": query, "variables": {"input": payload}}
    )
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL ERR: {data['errors']}")
    return data["data"]["saveTemplate"]["id"]

def create_endpoint(template_id):
    payload = {
        "name": "sd-v109-final-test-reprod",
        "templateId": template_id,
        "gpuTypeIds": ["NVIDIA GeForce RTX 5090"],
        "gpuCount": 2,
        "workersMin": 0,
        "workersMax": 5,
        "idleTimeout": 10,
        "networkVolumeId": "uomx1tyts8",
        "flashboot": True,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4
    }
    
    resp = requests.post("https://rest.runpod.io/v1/endpoints", headers=headers, json=payload)
    if not resp.ok:
         print(resp.text)
         if resp.status_code == 500:
             print("Retrying with even fewer fields...")
             payload.pop("networkVolumeId", None)
             payload.pop("flashboot", None)
             resp = requests.post("https://rest.runpod.io/v1/endpoints", headers=headers, json=payload)
    
    return resp.json()["id"]

if __name__ == "__main__":
    template_id = create_template()
    endpoint_id = create_endpoint(template_id)
    print(f"SUCCESS: New endpoint created with HF_TOKEN: {endpoint_id}")
