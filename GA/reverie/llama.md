# LLAMA implementation details
Okay, connecting to a local Ollama instance is a great way to avoid cloud rate limits and costs. The codebase already has some structure in place for a `llama` key type, which we can adapt for Ollama since it provides an OpenAI-compatible API endpoint.

1.  **File `utils.py`**
    * **Set Key Type:** Change `key_type = 'gemini'` to `key_type = 'llama'`.
    * **Configure API Base:** Set the `LLAMA_API_BASE` environment variable to your Ollama server address (usually `http://localhost:11434/v1`) or update the `llama_api_base_val` placeholder directly in `utils.py`. e.g export LLAMA_API_BASE=http://192.168.86.100:11434/v1
    * **Configure API Key:** Set the `LLAMA_API_KEY` environment variable or update the assignment in `utils.py`. For default Ollama setups, the key is often not required, so setting it to `"none"` or an empty string.
    * **Set Default Model:** Change `DEFAULT_LLAMA_CHAT_MODEL` to the specific model you have pulled and want to use in Ollama (e.g., `"llama3:8b"`, `"mistral:latest"`).

2.  * **Address Embeddings (`get_embedding`):** Ollama's OpenAI-compatible endpoint (`/v1/...`) usually doesn't support embeddings via `openai.Embedding.create`. Ollama has a separate endpoint (`/api/embeddings`). In my current setup I am hosting Ollama on an old machine that does the job, so I am offloading the network load on my dev machine which is newer. If you plan to use something remotely you'll have to update the code (more invasive). In future I might consider to refactor but for now that I barely know what I am touching it's fine...

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
        
        * 
### Some Popular Alternatives:

        * **`'all-mpnet-base-v2'`:** Another strong, widely used English model (slightly larger/slower than MiniLM but often higher quality).
        * **`'multi-qa-mpnet-base-dot-v1'`:** Optimized for semantic search / question-answering retrieval tasks.
        * **`'paraphrase-multilingual-mpnet-base-v2'`:** A good choice if you need multilingual support.
        * **`'BAAI/bge-large-en-v1.5'`:** (From BAAI on Hugging Face) - Often ranks very highly on the MTEB leaderboard, but is larger.
        * **`'thenlper/gte-large'`:** (From thenlper on Hugging Face) - Another high-performing model.
    

        
**note**: The `sentence-transformers` library will typically download the model automatically the first time you use it if it's not already cached locally.

## How to run Ollama on a different machine
1. By default it runs on the localhost port. To make it answers to all your LAN request use this `OLLAMA_HOST=0.0.0.0 ollama serve`
2. to test if it works: 

```sh 
curl http://192.168.1.100:11434/api/generate \
  -d '{"model": "llama3:8b", "prompt": "Hello!" }'
```
Since ollama serve is the API server, you can preload your model after starting the server with a simple one-liner:
ollama run llama3:8b --system "preload"

you can combine the broadcast and the load operation like this

```sh
kill $(lsof -ti :11434) 2>/dev/null || true
OLLAMA_HOST=0.0.0.0 ollama serve > /tmp/ollama.log 2>&1 &
sleep 5 && ollama run llama3:8b "hi"
```
