import requests
from typing import Dict

class ResearchClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def submit_experiment(self, config: Dict):
        resp = requests.post(f"{self.base_url}/experiments/run", json=config)
        return resp.json()

    def get_status(self, exp_id: str):
        resp = requests.get(f"{self.base_url}/experiments/status/{exp_id}")
        return resp.json()
