#!/bin/bash
set -e

echo "Starting LiveKit Server with Agent Dispatch Configuration..."
docker run -d \
  -p 7880:7880 \
  -p 7881:7881 \
  -p 7882:7882/udp \
  -e LIVEKIT_KEYS="devkey: secret" \
  -v /home/rajsrinivas/livekit/livekit.yaml:/etc/livekit.yaml \
  livekit/livekit-server \
  --config /etc/livekit.yaml

sleep 4

echo ""
echo "✅ Server started!"
echo ""
echo "Starting Agent..."
cd /home/rajsrinivas/livekit
source .livekit/bin/activate
python3 client.py start > agent.log 2>&1 &
AGENT_PID=$!

sleep 5

echo "✅ Agent started (PID: $AGENT_PID)"
echo ""
echo "Running Bot..."
timeout 75 python3 bot.py sample.wav 2>&1 | tail -50

echo ""
echo "🎉 Test complete!"
