import requests
import json

API = "http://127.0.0.1:5000/api"

session_id = "test_session"

r = requests.post(f"{API}/create_session", json={
    "session_id": session_id,
    "random_seed": 42,
    "systematic_noise_level": 0.10,
    "random_noise_level": 0.02
})
print("create_session:", r.status_code, r.text)

code = """\
experiment_design = []
experiment_design.append({'R': 50, 'Y': 50, 'B': 50, 'well': 'A1'})
experiment_design.append({'R': 80, 'Y': 40, 'B': 60, 'well': 'A2'})
print('Designed', len(experiment_design), 'experiments')
"""

r = requests.post(f"{API}/execute_batch", json={"session_id": session_id, "code": code})
print("execute_batch:", r.status_code)
print(json.dumps(r.json(), indent=2))

r = requests.post(f"{API}/download_results", json={"session_id": session_id})
print("download_results:", r.status_code)
if r.ok:
    open(f"color_results_{session_id}.csv", "wb").write(r.content)
    print("Saved CSV")
