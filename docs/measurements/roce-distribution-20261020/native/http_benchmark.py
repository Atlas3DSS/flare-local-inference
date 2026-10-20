import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root/'flare'))
from qualify.glm_performance import request

base = 'http://127.0.0.1:8001'
output = root/'logs/inference'
output.mkdir(exist_ok=False)

def get(path):
    with urllib.request.urlopen(base + path, timeout=15) as response:
        return json.load(response)

def save(name, value):
    (output/name).write_text(json.dumps(value, indent=2) + '\n')

models = get('/v1/models')
save('models.json', models)
save('health.json', get('/health'))
model = models['data'][0]['id']
cases = json.loads((root/'logs/glm-http-corpus.json').read_text())['cases']
for case in cases:
    case['request'].pop('top_k', None)
save('requests.json', cases)
results = []
for phase, selected in [('warmup', cases[:1]), ('batch1', cases[1:3]), ('batch4', cases[3:7])]:
    before = get('/metrics?format=json')
    start = time.monotonic()
    for case in selected:
        case['phase'] = phase
    def run(case):
        return request(base, case, native=True, model=model,
                       output=output/f"{phase}-{case['index']}.json")
    if phase == 'batch4':
        with ThreadPoolExecutor(max_workers=4) as pool:
            measured = list(pool.map(run, selected))
    else:
        measured = [run(case) for case in selected]
    result = {'phase': phase, 'wall_seconds': time.monotonic() - start,
              'requests': measured, 'metrics_before': before,
              'metrics_after': get('/metrics?format=json')}
    results.append(result)
    save('results.json', results)
    print(json.dumps({'phase': phase, 'wall_seconds': result['wall_seconds'],
                      'requests': len(measured)}), flush=True)
save('health-after.json', get('/health'))
