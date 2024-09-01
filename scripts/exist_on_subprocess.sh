while true; do
    for pid in "${pids[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            wait $pid
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "Process with PID $pid exited with code $exit_code"
                cleanup
                exit 1
            else
                echo "Process with PID $pid exited successfully"
                cleanup
                exit 0
            fi
        fi
    done
    sleep 1
done
