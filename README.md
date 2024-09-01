# module_combine_source

### Overview
This script is designed to merge audio signals from two microphones to enhance audio quality and sensitivity.
It accounts for time drift and employs echo cancellation to dynamically adjust time shifts for optimal audio
synchronization.



### System Requirements
- Python 3.10.12 or higher
- PulseAudio (compatible with standard versions)
- Dependencies as listed in `requirements.txt`

### Installation

#### Setting Up Python Environment
1. Ensure Python 3.10 is installed on your system.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
#### venv

$ python3 --version
Python 3.10.12
$ sudo apt install python3.10-venv
$ python3 -m venv venv

### Enable virtual env

source venv/bin/activate

## Install Python Deps

pip install -r requirements.txt

#### Configuration
The script uses several environment variables for configuration. Set these before running the script:

**Mandatory:**
- `RECORD_DEVICE_NAMES`: Comma-separated list of device names to record from. Example: `local_minus_fly_plate,local_usb_minus_fly_plate`

**Optional:**
- `CORRELATION_THRESHOLD`: Minimum threshold for correlation during synchronization. Default is `0.01`.
- `RECORD_RATE`: The sampling rate for recording. Default is `16000 Hz`.
- `RECORD_CHUNK_SIZE`: Size of the audio chunk for recording. Default is `2048`.
- `DECAY_HALF_LIFE_SEC`: The half-life in seconds for decay calculations. Default is `30`.
- `LAG_LIMIT_MS`: Maximum lag limit in milliseconds to adjust for time drift. Default is `750`.
- `MAX_SIZE`: Maximum buffer size for audio analyzing. Default is `65536`.

You can create a `.env` file or export them directly in your shell:

```bash
export RECORD_DEVICE_NAMES='local_minus_fly_plate,local_usb_minus_fly_plate'
export CORRELATION_THRESHOLD='0.02'
export RECORD_RATE='32000'
export RECORD_CHUNK_SIZE='8192'
export DECAY_HALF_LIFE_SEC='10'
export LAG_LIMIT_MS='280'
export MAX_SIZE='32768'
```

#### Device Configuration
List available sources using PulseAudio to identify correct device names:

```bash
pactl list short sources
```

### Usage
To run the script, ensure all environment variables are set as described above, then execute:

```bash
python module_combine_source.py
```

### Screw Detection and Clock Drift Management
To effectively manage clock drift and detect screw in initial setup:
1. **Initial Setup**: Use a loud sound or speak near the microphones to minimize drifting errors. Set `CORRELATION_THRESHOLD` to `0.02` and `DECAY_HALF_LIFE_SEC` to `5` for initial detection.
2. **Saving Configurations**: After detecting the screw, send a SIGTERM signal to the script to dump the settings to a file. Adjust the environment variables for more robust operation based on these settings.
3. **Restart Strategy**: It is beneficial to restart the script periodically (every N minutes) to reset the internal state and recalibrate the synchronization. Even with clock drift detection, it is challenging to maintain accurate slope indefinitely.
4. **Minimize `LAG_LIMIT_MS`**: Reducing this value is strongly recommended to enhance synchronization accuracy and reduce error margins.

### Troubleshooting
#### Common Issues
- **High CPU Usage**: Adjust the `RECORD_CHUNK_SIZE` and `DECAY_HALF_LIFE_SEC` to optimize performance.
- **Audio Sync Issues**: Ensure the `CORRELATION_THRESHOLD` and `LAG_LIMIT_MS` are set according to your environment specifics.
- **Device Connection Failures**: Verify the device names with `pactl list short sources` and ensure they are correctly specified in the `RECORD_DEVICE_NAMES`.

#### Error Handling
- The script is designed to terminate gracefully upon receiving SIGINT or SIGTERM signals, saving its state for future sessions.
- Errors related to device connection failures will raise exceptions and halt execution. Ensure device names are correct and devices are properly connected.

### Feedback
- Your feedback is crucial for improving this script. Please report any issues or suggestions for enhancements.

## Echo cancellation and noise reduction example

[voice-kodi-rt-deps.service](scripts/voice-kodi-rt-deps.service) is a systemd service that manages a script merging voice inputs.
It depends on PulseAudio and is configured to restart on failure for continuous operation.

[systemd_rt_dependencies.sh](scripts/systemd_rt_dependencies.sh) This script sets up a real-time voice merging application by:

- Creating necessary directories for logs and temporary files.
- Configuring PulseAudio and activating a Python virtual environment.
- Logging application output and starting the main script in the background.
- Continuously monitoring logs and gracefully stopping on exit.

[load_pulse_audio.sh](scripts/load_pulse_audio.sh) The main script for configuring PulseAudio - used by
- systemd_rt_dependencies.sh
- voice-kodi-rt-deps.service

`load_pulse_audio.sh` is an example script that configures PulseAudio for advanced audio processing tasks required by
the voice merging system. This script checks for the existence of required modules, loads necessary modules
if they are not present, and manages audio routing and processing settings dynamically based on system configuration
and current state.

#### Key Operations
- **Module Management**:
   - Dynamically loads various PulseAudio modules such as
     - `module-tunnel-source`
     - `module-tunnel-sink`
     - `module-loopback`
     - `module-null-sink`
     based on their absence to set up remote audio streaming and local echo cancellation.
- **Echo Cancellation**:
   - Loads the `module-echo-cancel` with specific configurations to reduce echo and improve audio clarity using advanced WebRTC methods.
   - Unloads existing echo cancellation modules if needed to reload them with updated configurations.
- **Remote Audio Handling**:
   - Handles remote audio streams by receiving them over UDP and routing them through PulseAudio using `gst-launch-1.0`.
   - Ensures that the remote audio proxy process is running and restarts it if necessary.

#### Script Configuration
- The script includes conditional checks to only load modules if they are not already loaded, thereby preventing
  redundant operations. And **can be easily restarted**.
- Error handling is incorporated using a `cleanup` function that gracefully stops related processes if errors occur during the execution of the script.

#### Logging
- The script outputs its operation logs to `./logs/remote_sound_proxy.log.log` and `./logs/merger.log.log`, providing detailed information on operations performed and their outcomes.
