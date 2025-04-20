#!/bin/bash
# ollama-service.sh
# Script to set up Ollama as a persistent service on macOS, accessible on LAN

# Get local IP address
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)

# Stop any running Ollama processes
pkill ollama || true
echo "Stopped any running Ollama instances"

# Set environment variables for LAN access
export OLLAMA_HOST="0.0.0.0:11434"
echo "Setting Ollama to listen on all interfaces at port 11434"
echo "Your LAN address is: $LOCAL_IP:11434"

# Start Ollama in the background
echo "Starting Ollama as a background process..."
nohup ollama serve > ~/ollama.log 2>&1 &

# Save the PID for later reference
echo $! > ~/ollama.pid
echo "Ollama is now running with PID: $(cat ~/ollama.pid)"
echo "Log file is available at: ~/ollama.log"

# Create a simple HTML status page
cat > ~/ollama_status.html << EOL
<!DOCTYPE html>
<html>
<head>
    <title>Ollama Status</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .running { background-color: #d4edda; color: #155724; }
        .stopped { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>Ollama Server Status</h1>
    <p>Server address: $LOCAL_IP:11434</p>
    <div id="status" class="status running">
        <h2>Status: Running</h2>
        <p>Ollama is currently running and accessible on your LAN.</p>
    </div>
    <h3>Usage Instructions:</h3>
    <ul>
        <li>From other devices on your network, use: <code>http://$LOCAL_IP:11434</code></li>
        <li>For API requests: <code>curl http://$LOCAL_IP:11434/api/tags</code></li>
    </ul>
</body>
</html>
EOL

echo "Created status page at ~/ollama_status.html"
echo "Ollama is now serving on your LAN at $LOCAL_IP:11434"
echo "Other devices can connect to this address"
echo ""
echo "To stop Ollama service, run: kill \$(cat ~/ollama.pid)"