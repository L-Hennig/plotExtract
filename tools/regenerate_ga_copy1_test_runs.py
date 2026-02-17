import json
import os
import time
import urllib.request

BASE = r"c:\Users\Lucas\OneDrive\UCL Internship\Plot_Extract"
RUNS_ROOT = os.path.join(BASE, "plots", "synthetic", "G", "GA_copy1")
API = "http://127.0.0.1:5001"

PREFIX = "GA_copy1_png.pv2_prompt_13."
SUFFIX = ".key4.web"


def list_run_dirs():
    out = []
    for name in os.listdir(RUNS_ROOT):
        p = os.path.join(RUNS_ROOT, name)
        if os.path.isdir(p) and name.startswith(PREFIX) and name.endswith(SUFFIX):
            out.append(name)
    out.sort()
    return out


def post_json(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(path):
    with urllib.request.urlopen(API + path, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_once(payload, run_number):
    start = post_json("/v2/run_all", payload)
    task_id = start.get("task_id")
    if not task_id:
        raise RuntimeError(f"Could not start run {run_number}: {start}")

    print(f"Run {run_number} started task_id={task_id}", flush=True)

    while True:
        st = get_json("/task_status/" + task_id)
        status = st.get("status")
        progress = st.get("progress", "")
        if status in ("completed", "failed", "cancelled", "error"):
            print(f"Run {run_number} finished status={status}", flush=True)
            return task_id, status
        print(f"Run {run_number} status={status} progress={progress[:100]}", flush=True)
        time.sleep(2.0)


def main():
    before = set(list_run_dirs())
    print(f"Before count: {len(before)}", flush=True)

    payload = {
        "image": "synthetic/G/GA_copy1/GA_copy1.png",
        "prompt": "prompt_13",
        "articleInfo": "",
        "llm_provider": "mistral",
        "llm_model": "mistral-large-2512",
        "rate_limit_backoff": False,
        "runInterpolation": False,
        "runPointwise": False,
        "leftX": 0,
        "rightX": 100,
        "bottomY": 0,
        "topY": 100,
    }

    ids = []
    statuses = []
    for i in range(2):
        task_id, status = run_once(payload, i + 1)
        ids.append(task_id)
        statuses.append(status)

    after = set(list_run_dirs())
    new_dirs = sorted(after - before)
    print("Task IDs:", ids, flush=True)
    print("Statuses:", statuses, flush=True)
    print("New dirs:", new_dirs, flush=True)


if __name__ == "__main__":
    main()
