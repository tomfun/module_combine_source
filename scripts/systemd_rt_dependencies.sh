#!/usr/bin/env bash

# Function to clean up processes on exit
cleanup() {
    echo "Stopping all subprocesses..."
    set +e
    kill -SIGINT "$remote_sound_proxy"
    kill -SIGINT "$merger"
    sleep 1
    kill -SIGTERM "$tail"
    wait # Wait for all processes to stop
    echo "All subprocesses stopped."
}

# Set trap to call cleanup function on SIGINT (Ctrl-C)
trap cleanup SIGINT

set -e

mkdir -p logs cache samples

source ./load_pulse_audio.sh

source venv/bin/activate
export PYTHONUNBUFFERED=1
source <(sed 's/^/export /' .env)

echo '-------' >> ./logs/merger.log.log
date >> ./logs/merger.log.log

PLAY_DEVICE_NAME=merger_sink python src/app/module_combine_source.py >> ./logs/merger.log.log 2>&1 &
merger=$!

tail -f ./logs/merger.log.log ./logs/remote_sound_proxy.log.log &
tail=$!

pids[0]=$remote_sound_proxy
pids[1]=$merger
source ./exist_on_subprocess.sh
