# CHANGE LOG

Chronological order of what is pushed out.

---

🗓️ **Apr 19 2025**
Added documentation, tied up here and there the repo. Had the first good run but I am still facing timeouts to the llama apis. Likely because the server is on ethernet and the dev client is on Wifi. I improve the server configuration by making sure the machine wouldn't go to sleep and other small tweaks for my setup.

* FIXED: crash when the reasoning folder already existed.
* Better error messages for what I have encountered

🗓️ **Apr 16 2025**
Forked and start smashing my head on a sharp corner on why it's not working. The project originally was born only for OpenAI, then it got forked by other wise people to add Azure and somewhat llama. the latest didn't really work. So I clean up the code, moved out all api keys and adjusted the code to use `utils.py` as the central place for the global configuration.

* Setup a machine to run Ollama. I made a [service deamon](../environment/setup/ollama/) so that it stays up and running.

* Standardized the api keys management
* GEMINI: Added [Gemini](https://ai.google.dev) support
* GEMINI: Added some graceful shutdown for when [rate limit](https://ai.google.dev/gemini-api/docs/rate-limits) ceiling is hit
