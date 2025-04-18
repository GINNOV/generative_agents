# AGA in Generative Agents

<p align="center" width="100%">
<img src="../doc/pic/aga_in_ga.gif" width="30%" height="30%">
</p>

This work is based on [Generative Agents: Interactive Simulacra of Human Behavior](https://github.com/joonspk-research/generative_agents). Generative Agents provides a platform that simulates a virtual town with both front-end and back-end capabilities. For convenience and to reduce experiment time, I [forked](https://github.com/AffordableGenerativeAgents/Affordable-Generative-Agents?tab=readme-ov-file) a version that operates purely on the back-end. For more detailed information about the platform, please refer to the original repo [Generative Agents](https://github.com/joonspk-research/generative_agents).

## Good To Know
I am learning the codebase as I go, I didn't want to use the overly expensive OpenAI apis. I waited a few months that the original repo would update for other AI labs, it didn't happen so I did on my own.

# Fork Changes
* standardized the api keys
* added [gemini](https://ai.google.dev) support
* added some graceful shutdown for when [rate limit](https://ai.google.dev/gemini-api/docs/rate-limits) is hit
* the code was failing when the reasoning folder already existed, fixed
* improved some error messages to make sense when things don't work out

all those changes are in `gemini_integration` branch until I make sure that all is running as intended. The `main` branch is still the original fork so you can see the difference and rescue yourself (and me :-) if something is off.

## Preparation
To set up your environment, you will need to modify `utils.py` file that contains your LLM API key and download the necessary packages.

### Step 1. Update Utils File
set the LLM that you are going to use in the `utils.py` file. There are two of them, one in backend folder and another in the simulation folder.

**For gemini:**
In your shell profile (oh however, you like otherwise), define `GOOGLE_API_KEY` like this `export GOOGLE_API_KEY="your-key-here"`

### Step 2. Install requirements.txt
Install everything listed in the `requirements.txt` file.

```pip install -r requirements.txt```

## Running a Simulation
The back-end only version is in `reverie_offline.py`, you should run in the following format:

`
python reverie_offline.py -o <the forked simulation> -t <the new simulation> -s <the total run step>`

**like this**:
```bash
python reverie_offline.py -o base_the_ville_isabella_maria_klaus -t aga_3_person -s 17280
```
start with a very small number (like 300) so that you can make sure that everything works.

--
# Visualization
To visualize, you need to go through three steps: 
1. Complete a simulation
2. Compress the simulation data
3. Use the Front-end visualization.

## Step 1. Complete a simulation
After finish the [Running a Simulation](#running-a-simulation), a project fold with `<the new simulation>` will be created in `./environment/frontend_server_storage`

### Step 2. Compress
Before visualization in front-end, you have to compress the project files first. 

change the code in `./reverie/compress_sim_storage.py`

```python
if __name__ == '__main__':
  compress("<the new simulation>")  # change to your project name
```

Run the following command:

```bash
python compress_sim_storage.py
```

## Step 3. Front-end visualization
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

note: I don't know what I am doing. I enjoy this type of project and I decided to face palm myself with the unknown, learn something and share what I figure out. If you have advices, share away. Good luck.