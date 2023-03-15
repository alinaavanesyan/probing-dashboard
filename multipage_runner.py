import os
import random
import time
import uuid
from collections import defaultdict
from datetime import datetime
from enum import Enum
from subprocess import PIPE, STDOUT, Popen
from threading import Thread
from typing import Dict, List, Union

from fastapi import FastAPI, File, Form, UploadFile


class JobStatus(str, Enum):
    wip = 'wip'
    done = 'done'
    error = 'error'

procs = {}
procs_stdouts = dict()
procs_ports = []
ports_range=(8000,9000)


def _deal_with_stdout(p, stdout_so_far):
    pid = str(p.pid)
    for line in p.stdout:
        if pid not in stdout_so_far: stdout_so_far[pid] = []
        stdout_so_far[pid].append(line.decode("utf-8"))

def run_one_more_process(commands, stdouts_storage):    
    p = Popen(commands, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    t = Thread(target=lambda: _deal_with_stdout(p, stdouts_storage), daemon=True)
    t.start()
    return p


def collect_stdout_so_far(pid, stdouts_storage):
    stdout = stdouts_storage.get(pid, [])
    stdout_len = len(stdout)
    return [stdout.pop(0) for _ in range(stdout_len)]


def _proc_tick(procs: dict):
    for proc in procs.values():
        start_time = proc.setdefault("start_time", datetime.now().timestamp())
        working_time = datetime.now().timestamp() - start_time
        status = JobStatus.done if proc["proc"].poll() is not None else JobStatus.wip
        proc["logs"].extend(collect_stdout_so_far(proc["id"], procs_stdouts))
        proc["status"] = status


def get_free_port():
    global procs_ports
    global ports_range
    print(f"known pids", procs_ports)
    for port in range(*ports_range):
        if port not in procs_ports:
            print(f"returning {port}")
            return port

def get_dashboard(data):
    global procs_ports
    global procs_stdouts
    global procs

    port = get_free_port()
    commands = ["python3", "dash_newgraph.py", port]

    proc = run_one_more_process(commands, procs_stdouts)
    start_time = datetime.now().timestamp()
    proc_id = str(proc.pid)
    logs = []
    procs[proc_id] = {"data": data, "id": proc_id, "start_time": start_time, 
                      "logs": logs, "proc": proc, "command": commands,
                      "port":port, "url": f"http://Probing-Dashboard:{port}/dashboard"}
    procs_ports.append(port)
    _proc_tick(procs)
    result = procs[proc_id]
    print(result)
    return result


app = FastAPI(docs_url='/api/docs')

@app.get("/getDashboard")
def probing_start(body):
    response = get_dashboard(body)
    return response
