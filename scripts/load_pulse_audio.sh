env | grep -i PULSE

pactl list short sinks
pactl list short sources
systemctl --user status pulseaudio --no-pager || echo 'pulseaudio dead!'

set +e

pactl list short modules | grep 'source_name=remote_pc_hdmi_monitor' \
  || pactl load-module module-tunnel-source source_name=remote_pc_hdmi_monitor server=192.168.0.108 source=alsa_output.pci-0000_01_00.1.hdmi-stereo-extra1.monitor

pactl list short modules | grep 'sink_name=remote_pc_hdmi' \
  || pactl load-module module-tunnel-sink sink_name=remote_pc_hdmi server=192.168.0.108 sink=alsa_output.pci-0000_01_00.1.hdmi-stereo-extra1

pactl list short modules | grep 'sink_name=remote_pc_compressor_hdmi' \
  || pactl load-module module-tunnel-sink sink_name=remote_pc_compressor_hdmi server=192.168.0.108 sink=eq_n_comp

#load-module module-combine-sink sink_name=local_n_remote_pc slaves=alsa_output.pci-0000_00_1f.3.analog-stereo,remote_pc_compressor_hdmi
pactl list short modules | grep -E 'module-loopback\s+source=remote_pc_hdmi.monitor sink=alsa_output.pci-0000_00_1f.3.analog-stereo' \
  || pactl load-module module-loopback source=remote_pc_hdmi.monitor sink=alsa_output.pci-0000_00_1f.3.analog-stereo \
    adjust_time=10 sink_dont_move=true source_dont_move=true latency_msec=200 \
  && pactl set-default-sink remote_pc_hdmi

set -e

pactl list short modules | grep 'sink_name=merger_sink' \
  || pactl load-module module-null-sink sink_name=merger_sink sink_properties='device.description="For_Merger_Sink"' rate=48000 \
  || cleanup

pactl list short modules | grep 'sink_name=null_sink' \
  || pactl load-module module-null-sink sink_name=null_sink sink_properties='device.description="Null_Sink"' rate=48000 \
  || cleanup

module_ids=($(pactl list short modules \
               | grep -E 'module-echo-cancel.+source_name=(local_usb_minus_remote_pc|local_minus_remote_pc)\s' \
               | cut -f 1))
if [ ${#module_ids[@]} -gt 0 ]; then
    for module_id in "${module_ids[@]}"; do
        echo "Unloading module: $module_id"
        pactl unload-module "$module_id"
    done
fi

pactl load-module module-echo-cancel use_master_format=1 aec_method=webrtc \
     aec_args="analog_gain_control=0\\ digital_gain_control=1\\ experimental_agc=1\\ noise_suppression=1\\ voice_detection=1\\ extended_filter=1\\ high_pass_filter=1" \
     source_master="alsa_input.usb-C-Media_Electronics_Inc._USB_PnP_Sound_Device-00.analog-mono" \
     source_name=local_usb_minus_remote_pc sink_master="null_sink" sink_name=remote_pc_speaker_usb \
  || cleanup

pactl load-module module-echo-cancel use_master_format=1 aec_method=webrtc \
     aec_args="analog_gain_control=0\\ digital_gain_control=1\\ experimental_agc=1\\ noise_suppression=1\\ voice_detection=1\\ extended_filter=1\\ high_pass_filter=1" \
     source_master="alsa_input.pci-0000_00_1f.3.analog-stereo" \
     source_name=local_minus_remote_pc     sink_master="null_sink" sink_name=remote_pc_speaker_lcl \
  || cleanup

pactl list short modules | grep for_remote_pc \
  || pactl load-module module-combine-sink sink_name=for_remote_pc \
     slaves=remote_pc_speaker_lcl,remote_pc_speaker_usb rate=48000 channels=2 \
  || cleanup

echo '-------' >> ./logs/remote_sound_proxy.log.log
date >> ./logs/remote_sound_proxy.log.log

# Manually listen for remote PCM audio via udp to cut it off from mics

remote_sound_proxy=$(cat ./logs/remote_sound_proxy.pid)
#/usr/bin/gst-launch-1.0 udpsrc port=26863 buffer-size=1500 mtu=1500 \
#  ! application/x-rtp,media=audio, encoding-name=MPA, payload=96, clock-rate=48000 \
#  ! rtpmpadepay ! queue ! mpegaudioparse ! mpg123audiodec \
#  ! pulsesink device=for_remote_pc | tee -a ./logs/remote_sound_proxy.log.log & rtpgstdepay
if ! kill -0 $remote_sound_proxy; then
  gst-launch-1.0 -v udpsrc port=26863   \
    caps="application/x-rtp,media=(string)audio, channels=(int)2, clock-rate=(int)48000, encoding-name=(string)L16" \
    ! rtpL16depay ! audioconvert ! queue ! pulsesink device=for_remote_pc \
    | tee -a ./logs/remote_sound_proxy.log.log &
  remote_sound_proxy=$!
  echo "$remote_sound_proxy" > ./logs/remote_sound_proxy.pid
fi

