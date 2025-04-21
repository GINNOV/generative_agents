# AGA in Generative Agents

<p align="center" width="100%">
<img src="./docs/images/aga_in_ga.gif" >
</p>

# Latest Changes Highlights
* standardized the api keys management
* added [gemini](https://ai.google.dev) support
* added some graceful shutdown for when [rate limit](https://ai.google.dev/gemini-api/docs/rate-limits) ceiling is hit
* crash when the reasoning folder already existed, fixed
* improved error messages

more details specific to this environment can be found [here](./docs/changelog.md).

## Preparation
1. cd to the root e.g. `gem-Generative-Agents/`
2. run `pip install -r requirements.txt`
3. cd to folder `gem-Generative-Agents/GA/reverie/backend_server`
4. configure based on which model you want to leverage (see below)
5. make a short run to test things `clear & python reverie_offline.py -o base_the_ville_isabella_maria_klaus -t cippa_person -s 50`

If you plan to use [LLama](./docs/llama.md) or Gemini Flash there's a read me that specifically gives guidance on those. Either way you either set environment variables or directly modify `utils.py` file that contains your LLM API key. then install the necessary packages.

### STEP 1: Update Utils File
set the LLM that you are going to use in the `utils.py` file by modifying `key_type = 'llama'`. Everything else should be already all set for a firt run.
**for gemini**
`export GOOGLE_API_KEY="your-key-here"`
**for llama**
`export OPENAI_API_KEY="your-key-here"`

### STEP 2: Install requirements
 * Create a virtual environment `python -m venv crazy_town-env`. Activate the environment
 * Install requirements ```pip install -r requirements.txt```

## Running a Simulation
The back-end only version is in `reverie_offline.py`, you should run in the following format:

`python reverie_offline.py -o <baseline_simulation> -t <new simulation> -s <the total run step>`

**like this**:

```bash
clear & python reverie_offline.py -o base_the_ville_isabella_maria_klaus -t nice_person -s 17280
```
**advice**: Start with a very small number (like 300) so that you can make sure that everything works before waiting for hours. I also suggest you dump the output of the console in to a log.txt to make sure you haven't missed logging errors. (keys, timeouts and so on). A QTT service would be nice, maybe in future.

---

# Visualization
To visualize, you need to go through three steps: 
1. Complete a simulation
2. Compress the simulation data
3. Use the Front-end visualization.

## STEP 1: Complete a simulation
After finish the [Running a Simulation](#running-a-simulation), a project fold with `<the new simulation>` will be created in `./environment/frontend_server_storage`

### STEP 2: Compress
Before visualization in front-end, you have to compress the project files first. 

change the code in `./reverie/compress_sim_storage.py`

```python
if __name__ == '__main__':
  compress("<the new simulation>")
```

Run the following command:

```bash
python compress_sim_storage.py
```

## STEP 4: Front-end visualization
setting up the front-end, first navigate to `environment/frontend_server` and run:
```bash
python manage.py runserver
```

To start the visualization, go to the following address on your browser: `http://localhost:8000/demo/<the new simulation>/<starting-time-step>/<simulation-speed>`. Note that `<the new simulation>` denote the same things as mentioned above. `<simulation-speed>` can be set to control the demo speed, where 1 is the slowest, and 5 is the fastest. For instance, visiting the following link will start a pre-simulated example, beginning at time-step 1, with a medium demo speed:  
[http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/](http://localhost:8000/demo/July1_the_ville_isabella_maria_klaus-step-3-20/1/3/)

# Experiment

The **Lifestyle policy** and **Social Impression Memory** are enabled by default. For ablation study, you can turn them off by:
```bash
python reverie_offline.py ... \
    --disable_policy \     # Turn off the Lifestyle policy
    --disable_relationship # Turn off the Social Impression Memory
```
# Storage
All relevant records for the experiments are generated in the `<project fold>/metrics`:

```
─ metrics
├── detail_info.json                # Complete LLM call log
├── function_name_count.json        # LLM function call count statistics.
├── function_name_fail_count.json   # LLM function call failure count statistics.
├── function_name_fail_reason.json  # LLM function call failure count statistics.
├── function_name_time.json         # Statistics on the reasons for LLM function call failures.
├── function_name_token.json        # Statistics on LLM token consumption by different functions.
├── model_count.json                # LLM model call count statistics.
├── model_token.json                # Statistics on LLM token consumption by different models.
└── <personas_name>.json            # All action logs for the corresponding agent with <personas_name>
```

During experiments, generated policies and embedding features are saved in `./environment/fribtebd_server/storage/public`. The number of policies will affect the token consumption of the experiment.

The implementation of **MindWandering** is in the branch of `mindwandering` of the forked repo.