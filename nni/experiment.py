import json
import yaml
import subprocess

def update_config(name, pid):
    with open('config.yaml') as f:
        doc = yaml.safe_load(f)
    doc['experimentName'] = name + "_" + str(pid)
    doc['searchSpaceFile'] = f"../params/{name}_{pid}.json"
    with open(f'configs/{name}_{pid}.yaml', 'w') as f:
        yaml.safe_dump(doc, f, default_flow_style=False)

pids = range(57)

experiment = {
    "name": "TM",
    "pids": pids,
    "optimize": ["T", "M"],
    "shared": ["M"],
    "default": {"T": 0.3, "M": 0.2, "tau": 0, "delta": 0}
}

for pid in experiment['pids']:
    params = {}
    params["pid"] = {"_type": "choice", "_value": [pid]}
    params["T"] = {"_type":"quniform","_value":[0.1, 0.5, 0.01]} if "T" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["T"]]}
    params["M"] = {"_type":"quniform","_value":[0.1, 0.3, 0.01]} if "M" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["M"]]}
    params["tau"] = {"_type":"quniform","_value":[0.01, 0.1, 0.01]} if "tau" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["tau"]]}
    params["delta"] = {"_type":"quniform","_value":[0.1, 1.0, 0.1]} if "delta" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["delta"]]}

    with open(f"params/{experiment['name']}_{pid}.json", "w") as outfile:
        json.dump(params, outfile)

    update_config(experiment['name'], pid)

    cmd_str = f"nnictl create --config configs/{experiment['name']}_{pid}.yaml --port {8080+pid}"
    subprocess.run(cmd_str, shell=True)