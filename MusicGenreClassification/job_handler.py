import os
import sys
import subprocess
import shlex
from collections import namedtuple
from queue import Queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait

from flask import Flask, request

app = Flask(__name__)
task_queue = Queue()
tasks = []
STATUS_SUBMITTED = 0x0
STATUS_STARTED = 0x1
STATUS_RUNNING = 0x2
STATUS_FINISHED = 0x3
STATUS_CANCELLED = 0x4


class Task:
    def __init__(self, submit_time=None, start_time=None, finish_time=None,
                 process=None, cmd=None):
        self.submit_time = submit_time
        self.start_time = start_time
        self.finish_time = finish_time
        self.process = process
        self.cmd = cmd
        self.status = STATUS_SUBMITTED


def worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        if task.status == STATUS_CANCELLED:
            continue
        cmd = shlex.split(task.cmd)
        p = subprocess.Popen(cmd)
        task.start_time = datetime.now()
        task.process = p
        task.status = STATUS_RUNNING
        p.wait()
        if task.status == STATUS_CANCELLED:
            continue
        task.status = STATUS_FINISHED
        task.finish_time = datetime.now()


@app.route('/', methods=['GET', 'POST'])
def submit_job():
    if request.method != 'POST':
        return render_index()
    cmd = request.form.get('cmd', None)
    if cmd is None or cmd == '':
        return render_index('No command.')
    task = Task(submit_time=datetime.now(), cmd=cmd)
    tasks.append(task)
    task_queue.put(task)
    return render_index('Submission succeed.')


@app.route('/cancel_job')
def cancel_job():
    idx = request.args.get('idx')
    if idx is None or not idx.isdigit():
        return render_index('Invalid index.')
    idx = int(idx)
    if idx < 0 or idx >= len(tasks):
        return render_index('Invalid index.')
    task = tasks[idx]
    if task.status in (STATUS_CANCELLED, STATUS_FINISHED):
        return render_index('Already finished.')
    task.status = STATUS_CANCELLED
    task.finish_time = datetime.now()
    if task.process is not None:
        task.process.terminate()
    return render_index('Job cancelled.')


def colored_text(text, color='black', bg_color='white'):
    return '<span style="background-color:{}; color: {};">{}</span>'.format(
        bg_color, color, text)

def format_status(task, idx):
    if task.status == STATUS_FINISHED:
        status = colored_text('Finished', 'white', 'green') 
        action = ''
    elif task.status == STATUS_RUNNING:
        status = colored_text('Running', 'white', 'blue')
        action = '(<a href="/cancel_job?idx={}">Cancel</a>)'.format(idx)
    elif task.status == STATUS_CANCELLED:
        status = colored_text('Cancelled', 'white', 'black')
        action = ''
    elif task.status == STATUS_SUBMITTED:
        status = colored_text('Pending', 'black', 'yellow')
        action = '(<a href="/cancel_job?idx={}">Cancel</a>)'.format(idx)
    else:
        raise ValueError('Invalid status')
    submit_time = task.submit_time.strftime('%Y-%m-%d %H:%M:%S')
    if task.start_time:
        start_time = task.start_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        start_time = '[Not yet started]'
    if task.finish_time:
        finish_time = task.finish_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        finish_time = '[Not yet finished]'
    fmt = ('Status: {:15s} Submitted at: {:25s} '
           'Started at: {:25s} Finished at: {:25s} Command: {} {}')
    return fmt.format(status, submit_time, start_time, finish_time,
                      task.cmd, action)


def render_job_status_page(status):
    return '''
<html>
    <head>
        <title>Job Status</title>
    </head>
    <body>
        <a href="/">Back</a>
    </body>
</html>
    '''.format(status)


def render_index(msg=''):
    status = [format_status(task, idx) for idx, task in enumerate(tasks)]
    status = ''.join('<li>{}</li>'.format(x) for x in status)
    return '''
<html>
    <head>
        <title>Submit Job</title>
    </head>
    <body>
        <p>{}</p>
        <form action="/" method="POST">
            <input type="text" name="cmd" placeholder="Command to execute" style="width:90%">
            <button type="submit">Submit</button>
        </form>
        <p>Recent Jobs:</p>
        <ul>{}</ul>
    </body>
</html>
    '''.format(msg, status)


with ThreadPoolExecutor(1) as executor:
    executor.submit(worker)
    port = int(sys.argv[1]) if len(sys.argv) == 2 else 7999
    app.run('0.0.0.0', port=port)
