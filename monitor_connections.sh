#!/bin/bash

echo "Monitoring connections to port 9700..."
echo "Press Ctrl+C to stop"

while true; do
    count=$(netstat -an | grep :9700 | grep ESTABLISHED | wc -l)
    timestamp=$(date '+%H:%M:%S')
    echo "$timestamp: $count connections to port 9700"
    sleep 1
done
