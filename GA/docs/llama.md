# LLAMA implementation details
Connecting to a local Ollama instance is a great way to avoid cloud rate limits and costs. The codebase already has some structure in place for a `llama` key type, which we can adapt for Ollama since it provides an OpenAI-compatible API endpoint. There are two things of being aware here:
* How to [setup](#how-to-run-ollama-on-a-different-machine) you Ollama server
* [Embeddings](#about-the-embedding-model-choice) management

**File `utils.py`**
    1. **Set Key Type:** Change `key_type = 'gemini'` to `key_type = 'llama'`.
    2. **Configure API Base:** Set the `LLAMA_API_BASE` environment variable to your Ollama server address (usually `http://localhost:11434/v1`) or update the `llama_api_base_val` placeholder directly in `utils.py`. e.g export LLAMA_API_BASE=http://192.168.86.100:11434/v1
    3. **Configure API Key:** Set the `LLAMA_API_KEY` environment variable or update the assignment in `utils.py`. For default Ollama setups, the key is often not required, so setting it to `"none"` or an empty string.
    4. **Set Default Model:** Change `DEFAULT_LLAMA_CHAT_MODEL` to the specific model you have pulled and want to use in Ollama (e.g., `"llama3:8b"`, `"mistral:latest"`).


 **File `gpt_structure.py`** 
 **Embeddings (`get_embedding`):** Ollama's OpenAI-compatible endpoint (`/v1/...`) usually doesn't support embeddings via `openai.Embedding.create`. Ollama has a separate endpoint (`/api/embeddings`). In my current setup I am hosting Ollama on an old machine that does the job, so I am offloading the network load on my dev machine which is newer. If you plan to use something remotely you'll have to update the code (more invasive). In future I might consider to refactor but for now that I barely know what I am touching it's fine...


```python
# local_embedding_model_name = 'all-MiniLM-L6-v2' # Original
local_embedding_model_name = 'BAAI/bge-large-en-v1.5' # Example using a different model
``` 

If you run the Ollama on the same machine it will bypass the embeddings by using `llm_request` function otherwise it will call `get_embedding` function which uses the sentence-transformers approach.



## About the embedding model choice

1.  **Where did `'all-MiniLM-L6-v2'` come from?**
    * This specific name (`'all-MiniLM-L6-v2'`) refers to a popular pre-trained model designed for sentence embeddings. It's made available through the **`sentence-transformers` library**, which itself leverages models often hosted on the Hugging Face Hub.
    * It was likely chosen as the default in the code because it offers a good balance:
        * **Fast:** It's relatively small compared to large generation models.
        * **Good Performance:** It performs well on many common embedding tasks like semantic search and similarity comparison.
        * **Widely Used:** It's a common baseline or starting point.

2.  **What other options are there?**
    * The `sentence-transformers` library gives you access to hundreds of pre-trained embedding models. You can choose based on your needs (performance, speed, language support, etc.).
    * **How to find models:**
        * **Sentence Transformers Docs:** The official documentation lists many recommended models: [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html)
        * **MTEB Leaderboard:** The Massive Text Embedding Benchmark (MTEB) leaderboard on Hugging Face ranks models based on performance across various tasks: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
        
### Some Popular Alternatives:

* **`'all-mpnet-base-v2'`:** Another strong, widely used English model (slightly larger/slower than MiniLM but often higher quality).
* **`'multi-qa-mpnet-base-dot-v1'`:** Optimized for semantic search / question-answering retrieval tasks.
* **`'paraphrase-multilingual-mpnet-base-v2'`:** A good choice if you need multilingual support.
* **`'BAAI/bge-large-en-v1.5'`:** (From BAAI on Hugging Face) - Often ranks very highly on the MTEB leaderboard, but is larger.
* **`'thenlper/gte-large'`:** (From thenlper on Hugging Face) - Another high-performing model.


        
**note**: The `sentence-transformers` library will typically download the model automatically the first time you use it if it's not already cached locally.

## How to run Ollama on a different machine
1. By default it runs on the localhost port. To make it answers to all your LAN requests use this script `ollama-service.sh` it's made for macOS.
2. to test if it works: 

    ```sh 
    curl http://192.168.1.100:11434/api/generate \
    -d '{"model": "llama3:latest", "prompt": "Hello!" }'
    ```

# Setting Up Ollama to Serve Indefinitely on Your LAN

These instructions will help you configure Ollama to run as a persistent service on your macOS system and make it accessible to other devices on your local network. I don't expect your LAN to be of such interest to some bad actors but if you feel sensitive about, take some steps for a better security layer than those steps.

## Overview

## Step 1: Save and Run the Service Script

1. Save the "Ollama LAN Service Setup" script to your home directory as `ollama-service.sh`
2. Make it executable:
   ```bash
   chmod +x ~/ollama-service.sh
   ```
3. Run the script to start Ollama:
   ```bash
   ~/ollama-service.sh
   ```

This script will:
- 🧨 Stop any running Ollama instances
- Configure Ollama to listen on all network interfaces
- Start Ollama as a background process
- Create a status page with your LAN IP address on your home root

## Step 2: Set Up Automatic Startup (Option 1 - Launch Agent)

1. Save the launch agent configuration to your LaunchAgents directory:
   ```bash
   mkdir -p ~/Library/LaunchAgents
   ```
2. Save the "Ollama Launch Agent Configuration" as `~/Library/LaunchAgents/com.user.ollama.plist`
3. Load the launch agent:
   ```bash
   launchctl load ~/Library/LaunchAgents/com.user.ollama.plist
   ```

This will ensure Ollama starts automatically when you log in and keeps running in the background.

## Step 2 Alternative: Set Up Automatic Startup (Option 2 - Crontab)

If you prefer using crontab instead of launch agents:

1. Open your crontab for editing:
   ```bash
   crontab -e
   ```
2. Add the following line to run the service script at startup:
   ```
   @reboot ~/ollama-service.sh
   ```
3. Save and exit

## Step 3: Test Your Setup

1. Check if Ollama is running:
   ```bash
   ps aux | grep ollama
   ```

2. Test the API locally:
   ```bash
   curl http://localhost:11434/api/tags
   ```
3. Open the status page in your browser:
   ```bash
   open ~/ollama_status.html
   ```

## Step 4: Connect from Other Devices

Other devices on your LAN can now connect to Ollama using your Mac's IP address:
- API endpoint: `http://YOUR_MAC_IP:11434`
- Example API call: `curl http://YOUR_MAC_IP:11434/api/tags`

## Troubleshooting

If Ollama isn't accessible on your LAN:

1. Check your firewall settings and ensure port 11434 is allowed
2. Verify Ollama is running with the correct host setting:
   ```bash
   ps aux | grep OLLAMA_HOST
   ```
3. Check the log files:
   ```bash
   cat ~/ollama.log
   cat ~/Library/Logs/ollama.log
   cat ~/Library/Logs/ollama-error.log
   ```

## Managing Your Ollama Service

- To stop Ollama manually:
  ```bash
  kill $(cat ~/ollama.pid)
  ```
- To unload the launch agent:
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.user.ollama.plist
  ```
- To reload after changes:
  ```bash
  launchctl unload ~/Library/LaunchAgents/com.user.ollama.plist
  launchctl load ~/Library/LaunchAgents/com.user.ollama.plist
  ```

## Security Considerations

This setup makes your Ollama instance accessible to all devices on your local network. For better security:

1. Consider adding authentication if possible
2. Only use this on trusted networks
3. Monitor the logs regularly for unusual access

I didn't because my closest neighbor is several miles away and they don't know to use the wifi 😅