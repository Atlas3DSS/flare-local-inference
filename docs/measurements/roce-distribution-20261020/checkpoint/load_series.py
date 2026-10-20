"""Run real checkpoint loading with independently timed process lifetimes."""
import json
from pathlib import Path
import subprocess
import sys
import time

root, rank = Path(sys.argv[1]), int(sys.argv[2])
for label, streams in [('load8', 8), ('load32', 32), ('load64', 64), ('load8-repeat', 8)]:
    directory = root / 'logs' / label
    directory.mkdir(exist_ok=False)
    unit = f'flare-glm53-{label}-20261020'
    command = ['systemd-run', '--user', '--unit='+unit,
        '--property=WorkingDirectory='+str(root/'flare'),
        '--property=StandardOutput=append:'+str(directory/'load.log'),
        '--property=StandardError=append:'+str(directory/'load.log'),
        '--setenv=PATH='+str(root/'flare/flare/target/release')+':/home/nvidia/.local/bin:/usr/local/bin:/usr/bin:/bin',
        '/usr/bin/time', '-v', '-o', str(directory/'load-time.txt'),
        str(root/'flare/.venv/bin/python'), '-u', str(root/'setup/load_glm_weights.py'),
        '--root',str(root),'--rank',str(rank),'--streams',str(streams),
        '--rendezvous','10.137.203.173:29630']
    subprocess.run(command, check=True)
    samplers = []
    with (directory/'sampler.log').open('w') as error:
        for target, name in [(unit+'.service','resources.jsonl'),
                            ('flare-glm53-agent-20261020.service','agent-resources.jsonl')]:
            output = (directory/name).open('w')
            samplers.append((subprocess.Popen(['/usr/bin/python3','-u',str(root/'setup/sample.py'),
                target,'900','1',str(root)], stdout=output,stderr=error), output))
        started = time.monotonic()
        captured = False
        while True:
            state = dict(line.split('=',1) for line in subprocess.check_output(
                ['systemctl','--user','show',unit,'-p','ActiveState','-p','ExecMainStatus'],text=True).splitlines())
            if not captured and time.monotonic()-started > 40:
                (directory/'sockets.txt').write_text(subprocess.check_output(['ss','-tinp'],text=True))
                captured=True
            if state['ActiveState'] not in ('active','activating'):
                break
            if time.monotonic()-started > 900:
                raise TimeoutError(label)
            time.sleep(2)
        for process, output in samplers:
            if process.poll() is None:
                process.terminate()
            process.wait()
            output.close()
    (directory/'exit.json').write_text(json.dumps(state)+'\n')
    print(json.dumps(dict(label=label,rank=rank,**state)),flush=True)
    if state['ExecMainStatus'] != '0':
        raise RuntimeError(f'{label} failed: {state}')
