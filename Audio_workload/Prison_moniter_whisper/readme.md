**Quick Start — UltraEdge Whisper**  
**0. Download Whisper Models (One-time setup on UltraEdge)**  
# Create models directory  
 sudo rm -rf /var/lib/ultraedge/whisper-store  
 sudo mkdir -p /var/lib/ultraedge/whisper-store/models  
   
 # Download models using Docker  
 docker run -it \  
   --entrypoint /bin/bash \  
   -v /var/lib/ultraedge/whisper-store/models:/models \  
   ghcr.io/ggml-org/whisper.cpp:main \  
   -c "  
     ./models/download-ggml-model.sh base /models &&  
     ./models/download-ggml-model.sh base.en /models  
   "  
   
This downloads the Whisper models. Run this once before starting the server.  
**1. Start Whisper Server (On UltraEdge machine)**  
IMAGE="864899848864.dkr.ecr.ap-south-1.amazonaws.com/ue/sp:whisper_GPU_Latest"  
 sudo -E ./target/debug/ultraedge \  
   --root /var/lib/ultraedge run \  
   --gpus all \  
   --image "$IMAGE" \  
   --mount /var/lib/ultraedge/whisper-store/models:/models:rw \  
   --publish 8062:8062 \  
   -- \  
   /bin/sh -lc 'exec /app/build/bin/whisper-server \  
   -m /models/ggml-base.en.bin \  
   --host 0.0.0.0 \  
   --port 8062'  
   
This starts the Whisper server on port 8062. Leave this terminal running.  
**2. Setup on Host Machine**  
# Install Python packages  
 pip install requests numpy ffmpeg-python  
   
 # Clone the project  
 git clone <your-repo>  
 cd prison-monitor  
   
 # Create directories  
 mkdir -p data/cells config results  
   
 # Create config files  
 echo "escape\nweapon\nkill" > config/threat_words.txt  
 echo "" > config/profanity.txt  
   
**3. Test Connection**  
# Place a test audio file in data/cells/  
 cp /path/to/test_audio.wav data/cells/test.wav  
   
 # Test with curl (from host machine)  
 curl -X POST \  
   -F "file=@data/cells/test.wav" \  
   http://localhost:8062/inference  
   
**4. Run Monitoring**  
**Monitor single file:**  
python cli.py --whisper-host localhost --whisper-port 8062 monitor CELL-001 data/cells/test.wav  
   
**Batch process directory:**  
python cli.py \  
  --whisper-host localhost \  
  --whisper-port 8062 \  
  sweep data/cells \  
  --cycles 36 \  
  --json-out results/output.json  
   
   
**5. View Results**  
# Pretty print JSON  
 cat results/output.json | python -m json.tool  
   
 # Check flagged cells  
 jq '.flagged_cells' results/output.json  
   
 # Get all detections  
 jq '.results[] | select(.is_flagged == true)' results/output.json  
   
**Environment Variables (Optional)**  
# Use environment variables instead of CLI args  
 export WHISPER_HOST=192.168.1.100  
 export WHISPER_PORT=8062  
   
 # Modify cli.py to read these (or keep using --whisper-host flag)  
   
**Common Commands**  
# Tamil language  
 python cli.py --language ta monitor CELL-001 audio.wav  
   
 # Multiple threads (process different folders)  
 python cli.py sweep data/cells/block_a --json-out results/block_a.json &  
 python cli.py sweep data/cells/block_b --json-out results/block_b.json &  
   
 # Continuous monitoring  
 while true; do  
   python cli.py sweep data/cells/incoming --json-out results/new_sweep.json  
   sleep 60  
 done  
   
**Troubleshooting**  
# Test if Whisper server is running  
 curl http://localhost:8062/inference  
   
 # Check if port 8062 is listening  
 netstat -tlnp | grep 8062  
   
 # Install ffmpeg if needed  
 sudo apt-get install ffmpeg  # Ubuntu  
 brew install ffmpeg          # macOS  
   
**That's it! 🎉**  
Your code now runs outside the container and talks to Whisper via HTTP on port 8062.  
See ULTRAEDGE_SETUP.md for detailed documentation.  
