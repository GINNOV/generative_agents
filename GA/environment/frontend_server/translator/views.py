"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: views.py
Modifications by: Gemini
"""
import os
import string
import random
import json
from os import listdir
import os
import logging # Added for better error logging

import datetime
from django.shortcuts import render, redirect, HttpResponseRedirect, HttpResponse # Added HttpResponse
from django.http import JsonResponse
# Assuming global_methods is accessible, adjust import if needed
# Note: global_methods.py exists in frontend_server folder based on user confirmation
try:
    # Use the local version first if available
    from .global_methods import check_if_file_exists, find_filenames
except ImportError:
    # Fallback if not found locally (adjust path if necessary)
    import sys
    # This path might need adjustment depending on exact structure and PYTHONPATH
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reverie', 'backend_server'))
    try:
        from global_methods import check_if_file_exists, find_filenames
    except ImportError:
        print("ERROR: Could not import global_methods from expected locations in views.py.")
        # Define dummy functions or raise error if critical
        def check_if_file_exists(path): return False
        def find_filenames(path, suffix): return []


from django.contrib.staticfiles.templatetags.staticfiles import static
from .models import * # models.py is empty, so this doesn't import anything specific

# Configure logging
# logging.basicConfig(level=logging.INFO) # Basic config might be done elsewhere in Django
logger = logging.getLogger(__name__) # Use Django's logging mechanism preferably

# --- landing view remains the same ---
def landing(request):
  context = {}
  # Assuming landing.html exists in templates/landing/
  template = "landing/landing.html"
  landing_template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', template)
  if not os.path.exists(landing_template_path):
       logger.warning(f"Template not found: {landing_template_path}. Redirecting to home.")
       return redirect('home') # Redirect to the improved home view
  return render(request, template, context)

# --- demo view remains the same (with added robustness) ---
def demo(request, sim_code, step, play_speed="2"):
  # Construct paths relative to the Django project base or use absolute paths
  # Assuming storage is relative to the directory containing manage.py
  manage_py_dir = os.path.dirname(os.path.abspath(__file__)) # views.py location
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # Go up to frontend_server dir

  storage_base = os.path.join(base_dir, 'compressed_storage')
  move_file = os.path.join(storage_base, sim_code, "master_movement.json")
  meta_file = os.path.join(storage_base, sim_code, "meta.json")
  persona_folder = os.path.join(storage_base, sim_code, "personas")

  # Basic existence check
  if not os.path.exists(move_file) or not os.path.exists(meta_file):
       logger.error(f"Compressed simulation data not found for sim_code: {sim_code} at {storage_base}")
       return HttpResponse(f"Error: Compressed simulation data not found for '{sim_code}'. Check folder name and ensure compression was successful.", status=404)

  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4,
                    "4": 8, "5": 16, "6": 32}
  play_speed = play_speed_opt.get(play_speed, 2) # Use get for default

  # Loading the basic meta information about the simulation.
  meta = dict()
  try:
    with open (meta_file) as json_file:
      meta = json.load(json_file)
  except Exception as e:
       logger.error(f"Error loading meta file {meta_file}: {e}")
       return HttpResponse(f"Error loading simulation metadata for '{sim_code}'.", status=500)

  sec_per_step = meta.get("sec_per_step", 10) # Default if missing
  start_date_str = meta.get("start_date", "January 1, 2023") # Default if missing
  try:
      start_datetime = datetime.datetime.strptime(start_date_str + " 00:00:00",
                                                  '%B %d, %Y %H:%M:%S')
      start_datetime += datetime.timedelta(seconds=sec_per_step * step)
      start_datetime_iso = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")
  except ValueError as e:
       logger.error(f"Error parsing date '{start_date_str}' from meta file: {e}")
       start_datetime_iso = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S") # Fallback

  # Loading the movement file
  raw_all_movement = dict()
  try:
    with open(move_file) as json_file:
      raw_all_movement = json.load(json_file)
  except Exception as e:
       logger.error(f"Error loading movement file {move_file}: {e}")
       return HttpResponse(f"Error loading simulation movement data for '{sim_code}'.", status=500)

  # Loading all names of the personas
  persona_names = []
  persona_names_set = set()
  # Get persona names from the first step's keys in movement data OR from persona folder
  first_step_key = "0" # Assuming step 0 always exists
  if first_step_key in raw_all_movement and raw_all_movement[first_step_key]:
       for p in list(raw_all_movement[first_step_key].keys()):
            initial = p[0] + p.split(" ")[-1][0] if " " in p else p[:2]
            persona_names.append({"original": p,
                                "underscore": p.replace(" ", "_"),
                                "initial": initial.upper()})
            persona_names_set.add(p)
  elif os.path.exists(persona_folder): # Fallback to persona folder names
       try:
            for p_folder in os.listdir(persona_folder):
                 if os.path.isdir(os.path.join(persona_folder, p_folder)) and not p_folder.startswith('.'):
                      p = p_folder # Assume folder name is persona name
                      initial = p[0] + p.split(" ")[-1][0] if " " in p else p[:2]
                      persona_names.append({"original": p,
                                         "underscore": p.replace(" ", "_"),
                                         "initial": initial.upper()})
                      persona_names_set.add(p)
       except Exception as e:
            logger.error(f"Error listing persona folders in {persona_folder}: {e}")
            # Continue without persona names if folder listing fails

  # <all_movement> is the main movement variable that we are passing to the
  # frontend. Whereas we use ajax scheme to communicate steps to the frontend
  # during the simulation stage, for this demo, we send all movement
  # information in one step.
  all_movement = dict()

  # Preparing the initial step.
  # <init_prep> sets the locations and descriptions of all agents at the
  # beginning of the demo determined by <step>.
  init_prep = dict()
  max_step_in_data = 0
  if raw_all_movement:
       digit_keys = [int(k) for k in raw_all_movement.keys() if k.isdigit()]
       if digit_keys:
            max_step_in_data = max(digit_keys)

  # Ensure requested step is not beyond available data
  step = min(step, max_step_in_data)

  # Find the state at the *exact* starting step requested
  start_step_str = str(step)
  if start_step_str in raw_all_movement:
       val = raw_all_movement[start_step_str]
       for p in persona_names_set:
            if p in val and "movement" in val[p] and val[p]["movement"] is not None:
                 init_prep[p] = val[p]
            elif p not in init_prep: # Provide default only if not set by an earlier step
                 init_prep[p] = {"movement": [0,0], "pronunciatio": "❓", "description": "Initial state", "chat": None}
  else: # If exact step missing, provide defaults for all
       logger.warning(f"Movement data for exact start step {step} not found in {move_file}. Using defaults.")
       for p in persona_names_set:
            init_prep[p] = {"movement": [0,0], "pronunciatio": "❓", "description": "Initial state", "chat": None}


  persona_init_pos = dict()
  for p_name_obj in persona_names:
       p_orig = p_name_obj["original"]
       p_under = p_name_obj["underscore"]
       if p_orig in init_prep and "movement" in init_prep[p_orig] and init_prep[p_orig]["movement"] is not None:
            persona_init_pos[p_under] = init_prep[p_orig]["movement"]
       else:
            persona_init_pos[p_under] = [0, 0] # Default to 0,0
            logger.warning(f"Initial position missing or invalid for {p_orig} at step {step}, using default.")


  all_movement[step] = init_prep # Store the state *at* the target start step

  # Finish loading <all_movement> from the target start step onwards
  for int_key in range(step + 1, max_step_in_data + 1):
    key = str(int_key)
    if key in raw_all_movement:
        all_movement[int_key] = raw_all_movement[key]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": json.dumps(persona_init_pos),
             "all_movement": json.dumps(all_movement),
             "start_datetime": start_datetime_iso, # Use ISO format
             "sec_per_step": sec_per_step,
             "play_speed": play_speed,
             "mode": "demo"}
  template = "demo/demo.html" # Use the specific demo template

  return render(request, template, context)

# --- UIST_Demo view remains the same ---
def UIST_Demo(request):
  # Consider making the sim_code dynamic or checking existence
  return demo(request, "March20_the_ville_n25_UIST_RUN-step-1-141", 2160, play_speed="3")


# --- MODIFIED home view ---
def home(request):
    # Define paths relative to the manage.py location
    manage_py_dir = os.path.dirname(os.path.abspath(__file__)) # views.py location
    base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # Go up to frontend_server dir

    # --- Scan for compressed simulations ---
    compressed_storage_dir = os.path.join(base_dir, "compressed_storage")
    available_compressed_sims = []
    if os.path.isdir(compressed_storage_dir):
        try:
            available_compressed_sims = sorted([
                d for d in os.listdir(compressed_storage_dir)
                if os.path.isdir(os.path.join(compressed_storage_dir, d)) and not d.startswith('.')
            ])
        except Exception as e:
            logger.error(f"Error scanning compressed_storage directory: {e}")
    else:
        logger.warning(f"Compressed storage directory not found: {compressed_storage_dir}")

    # --- Scan for uncompressed simulations (optional) ---
    uncompressed_storage_dir = os.path.join(base_dir, "storage")
    available_uncompressed_sims = []
    if os.path.isdir(uncompressed_storage_dir):
        try:
            # List directories, excluding hidden, 'public', and 'base_*'
            available_uncompressed_sims = sorted([
                d for d in os.listdir(uncompressed_storage_dir)
                if os.path.isdir(os.path.join(uncompressed_storage_dir, d))
                   and not d.startswith('.')
                   and d != 'public' # Exclude public folder explicitly
                   and not d.startswith('base_') # Exclude base folders
            ])
        except Exception as e:
            logger.error(f"Error scanning storage directory: {e}")
    else:
        logger.warning(f"Storage directory not found: {uncompressed_storage_dir}")


    # --- Check for live backend status ---
    # Use the presence of curr_sim_code.json in temp_storage as the indicator
    # temp_storage is sibling to frontend_server directory based on folder_structure.txt
    temp_storage_dir = os.path.join(base_dir, "temp_storage")
    f_curr_sim_code = os.path.join(temp_storage_dir, "curr_sim_code.json")
    live_backend_status = {
        "running": False,
        "sim_code": None,
        "step": None # Step might be harder to get reliably without backend running
    }
    if check_if_file_exists(f_curr_sim_code): # Use helper from global_methods
        try:
            with open(f_curr_sim_code) as json_file:
                data = json.load(json_file)
                live_backend_status["sim_code"] = data.get("sim_code")
                live_backend_status["running"] = bool(live_backend_status["sim_code"])
                # Optionally try to read step from curr_step.json if needed
                # f_curr_step = os.path.join(temp_storage_dir, "curr_step.json")
                # if check_if_file_exists(f_curr_step):
                #    with open(f_curr_step) as step_file:
                #        live_backend_status["step"] = json.load(step_file).get("step", 0)

        except Exception as e:
            logger.error(f"Error reading live backend status file {f_curr_sim_code}: {e}")
            live_backend_status["running"] = False # Assume not running if file is unreadable
    else:
         logger.info(f"Live backend status file not found: {f_curr_sim_code}")


    # --- Prepare Context ---
    context = {
        "available_compressed_sims": available_compressed_sims,
        "available_uncompressed_sims": available_uncompressed_sims,
        "live_backend": live_backend_status,
        "mode": "home" # Indicate this is the home/launcher page
    }
    template = "home/home.html" # Render the main home template

    return render(request, template, context)


# --- replay view remains the same ---
def replay(request, sim_code, step):
  # Similar logic to demo, but reads from storage/ instead of compressed_storage/
  # Needs error handling and path adjustments like demo view
  manage_py_dir = os.path.dirname(os.path.abspath(__file__)) # views.py location
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # Go up to frontend_server dir
  storage_base = os.path.join(base_dir, 'storage')
  sim_storage_path = os.path.join(storage_base, sim_code)
  persona_folder = os.path.join(sim_storage_path, "personas")

  if not os.path.isdir(sim_storage_path):
       logger.error(f"Uncompressed simulation data not found for sim_code: {sim_code}")
       return HttpResponse(f"Error: Uncompressed simulation data not found for '{sim_code}'.", status=404)

  step = int(step)

  persona_names = []
  persona_names_set = set()
  if os.path.isdir(persona_folder):
       try:
            for p_folder in os.listdir(persona_folder):
                 if os.path.isdir(os.path.join(persona_folder, p_folder)) and not p_folder.startswith('.'):
                      p = p_folder
                      initial = p[0] + p.split(" ")[-1][0] if " " in p else p[:2]
                      persona_names.append([p, p.replace(" ", "_"), initial.upper()]) # Add initial
                      persona_names_set.add(p)
       except Exception as e:
            logger.error(f"Error listing persona folders in {persona_folder}: {e}")
  else:
       logger.warning(f"Persona folder not found for replay: {persona_folder}")


  persona_init_pos = []
  environment_folder = os.path.join(sim_storage_path, "environment")
  file_count = []
  latest_step = 0
  if os.path.isdir(environment_folder):
       try:
            digit_files = [int(f.split(".")[0]) for f in os.listdir(environment_folder)
                           if f.endswith('.json') and f[:-5].isdigit()]
            if digit_files:
                 latest_step = max(digit_files)
                 file_count = digit_files # Keep the list if needed later
       except Exception as e:
            logger.error(f"Error listing environment files in {environment_folder}: {e}")

  # Ensure requested step is valid
  step = min(step, latest_step)
  curr_json_path = os.path.join(environment_folder, f'{step}.json') # Use requested step for init pos

  if os.path.exists(curr_json_path):
       try:
            with open(curr_json_path) as json_file:
                 persona_init_pos_dict = json.load(json_file)
                 for p_name_list in persona_names:
                      p_orig = p_name_list[0]
                      pos_data = persona_init_pos_dict.get(p_orig, {})
                      persona_init_pos.append([p_orig, pos_data.get("x", 0), pos_data.get("y", 0)])
       except Exception as e:
            logger.error(f"Error loading environment file {curr_json_path}: {e}")
            # Set default positions if file loading fails
            for p_name_list in persona_names: persona_init_pos.append([p_name_list[0], 0, 0])
  else:
       logger.warning(f"Initial environment file not found for step {step}: {curr_json_path}")
       for p_name_list in persona_names: persona_init_pos.append([p_name_list[0], 0, 0]) # Default positions


  # --- Load Meta for Replay ---
  # Assuming meta file might be in reverie/ or root of sim_storage
  meta_file = None
  meta_file_reverie = os.path.join(sim_storage_path, "reverie", "meta.json")
  meta_file_root = os.path.join(sim_storage_path, "meta.json")
  if os.path.exists(meta_file_reverie): meta_file = meta_file_reverie
  elif os.path.exists(meta_file_root): meta_file = meta_file_root

  meta = {}
  if meta_file:
       try:
            with open(meta_file) as json_f: meta = json.load(json_f)
       except Exception as e: logger.error(f"Error loading meta file for replay {meta_file}: {e}")

  sec_per_step = meta.get("sec_per_step", 10)
  start_date_str = meta.get("start_date", "January 1, 2023")
  try:
      start_datetime = datetime.datetime.strptime(start_date_str + " 00:00:00", '%B %d, %Y %H:%M:%S')
      start_datetime += datetime.timedelta(seconds=sec_per_step * step)
      start_datetime_iso = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")
  except ValueError as e:
      logger.error(f"Error parsing date '{start_date_str}' from meta file for replay: {e}")
      start_datetime_iso = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos, # Send list of lists [name, x, y]
             "start_datetime": start_datetime_iso, # Add time info
             "sec_per_step": sec_per_step,        # Add step duration
             "mode": "replay"} # Mode indicates it uses full storage
  template = "home/home.html" # Re-use home.html for replay UI? Needs verification.
  # Ensure home.html template can handle the context for 'replay' mode
  # It might need adjustments to load raw movement data instead of compressed
  return render(request, template, context)


# --- replay_persona_state view remains the same ---
def replay_persona_state(request, sim_code, step, persona_name):
  # Construct paths relative to the Django project base or use absolute paths
  manage_py_dir = os.path.dirname(os.path.abspath(__file__)) # views.py location
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # Go up to frontend_server dir
  storage_base = os.path.join(base_dir, 'storage')
  compressed_storage_base = os.path.join(base_dir, 'compressed_storage')

  sim_storage_path = os.path.join(storage_base, sim_code)
  compressed_storage_path = os.path.join(compressed_storage_base, sim_code)

  persona_name_underscore = persona_name.replace(" ", "_") # Ensure underscore version used internally if needed
  persona_name_orig = persona_name.replace("_", " ") # Ensure original name used for paths

  # Determine base memory path (prefer compressed if exists)
  memory_base = None
  compressed_memory_path = os.path.join(compressed_storage_path, "personas", persona_name_orig, "bootstrap_memory")
  uncompressed_memory_path = os.path.join(sim_storage_path, "personas", persona_name_orig, "bootstrap_memory")

  if os.path.exists(compressed_memory_path):
       memory_base = compressed_memory_path
       logger.info(f"Loading persona state from compressed: {compressed_memory_path}")
  elif os.path.exists(uncompressed_memory_path):
       memory_base = uncompressed_memory_path
       logger.info(f"Loading persona state from uncompressed: {uncompressed_memory_path}")
  else:
       logger.error(f"Persona memory folder not found for {persona_name_orig} in sim {sim_code}")
       return HttpResponse(f"Error: Persona data not found for '{persona_name_orig}' in simulation '{sim_code}'.", status=404)

  # Load memory files with error handling
  scratch = {}
  spatial = {}
  associative = {}
  try:
    scratch_path = os.path.join(memory_base, "scratch.json")
    if os.path.exists(scratch_path):
         with open(scratch_path) as json_file: scratch = json.load(json_file)
    else: logger.warning(f"scratch.json not found at {scratch_path}")
  except Exception as e: logger.error(f"Error loading scratch.json for {persona_name_orig}: {e}")

  try:
    spatial_path = os.path.join(memory_base, "spatial_memory.json")
    if os.path.exists(spatial_path):
         with open(spatial_path) as json_file: spatial = json.load(json_file)
    else: logger.warning(f"spatial_memory.json not found at {spatial_path}")
  except Exception as e: logger.error(f"Error loading spatial_memory.json for {persona_name_orig}: {e}")

  try:
    assoc_path = os.path.join(memory_base, "associative_memory", "nodes.json")
    if os.path.exists(assoc_path):
         with open(assoc_path) as json_file: associative = json.load(json_file)
    else: logger.warning(f"Associative memory nodes.json not found at {assoc_path}")
  except Exception as e: logger.error(f"Error loading associative_memory/nodes.json for {persona_name_orig}: {e}")


  a_mem_event = []
  a_mem_chat = []
  a_mem_thought = []

  # Sort associative keys numerically if possible, handle potential errors
  try:
       node_keys = sorted(associative.keys(), key=lambda x: int(x.split('_')[-1]) if x.startswith('node_') else float('inf'))
  except ValueError:
       node_keys = sorted(associative.keys()) # Fallback to string sort if key format is unexpected

  for node_id in reversed(node_keys): # Iterate newest first
    node_details = associative.get(node_id) # Use .get for safety
    if not node_details or not isinstance(node_details, dict): continue # Skip malformed nodes

    node_type = node_details.get("type")
    if node_type == "event":
      a_mem_event.append(node_details)
    elif node_type == "chat":
      a_mem_chat.append(node_details)
    elif node_type == "thought":
      a_mem_thought.append(node_details)

  context = {"sim_code": sim_code,
             "step": step,
             "persona_name": persona_name_orig, # Use original name for display
             "persona_name_underscore": persona_name_underscore, # Keep if template uses it
             "scratch": scratch,
             "spatial": spatial,
             "a_mem_event": a_mem_event,
             "a_mem_chat": a_mem_chat,
             "a_mem_thought": a_mem_thought}
  template = "persona_state/persona_state.html" # Ensure this template exists
  return render(request, template, context)


# --- path_tester view remains the same ---
def path_tester(request):
  context = {}
  template = "path_tester/path_tester.html" # Assuming this template exists
  path_tester_template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', template)
  if not os.path.exists(path_tester_template_path):
       logger.warning(f"Template not found: {path_tester_template_path}. Rendering basic response.")
       return HttpResponse("Path Tester page not fully configured.")
  return render(request, template, context)


# --- process_environment view remains the same ---
def process_environment(request):
  """
  <FRONTEND to BACKEND>
  Receives environment state from frontend and saves it.
  """
  manage_py_dir = os.path.dirname(os.path.abspath(__file__)) # views.py location
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # Go up to frontend_server dir
  storage_base = os.path.join(base_dir, 'storage')

  try:
      data = json.loads(request.body)
      step = data.get("step")
      sim_code = data.get("sim_code")
      environment = data.get("environment")

      if step is None or sim_code is None or environment is None:
           logger.error("Missing data in process_environment request.")
           return HttpResponse("Error: Missing data.", status=400)

      sim_env_path = os.path.join(storage_base, sim_code, "environment")
      os.makedirs(sim_env_path, exist_ok=True) # Ensure directory exists

      with open(os.path.join(sim_env_path, f"{step}.json"), "w") as outfile:
          json.dump(environment, outfile, indent=2)

      return HttpResponse("received")
  except json.JSONDecodeError:
       logger.error("Invalid JSON received in process_environment.")
       return HttpResponse("Error: Invalid JSON.", status=400)
  except Exception as e:
       logger.error(f"Error processing environment: {e}")
       return HttpResponse("Error processing request.", status=500)


# --- update_environment view remains the same ---
def update_environment(request):
  """
  <BACKEND to FRONTEND>
  Sends movement data for a specific step to the frontend.
  """
  manage_py_dir = os.path.dirname(os.path.abspath(__file__)) # views.py location
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # Go up to frontend_server dir
  storage_base = os.path.join(base_dir, 'storage')
  response_data = {"<step>": -1} # Default response

  try:
      data = json.loads(request.body)
      step = data.get("step")
      sim_code = data.get("sim_code")

      if step is None or sim_code is None:
           logger.error("Missing data in update_environment request.")
           return JsonResponse(response_data) # Return default error response

      move_file_path = os.path.join(storage_base, sim_code, "movement", f"{step}.json")

      if check_if_file_exists(move_file_path): # Use helper
          try:
               with open(move_file_path) as json_file:
                    response_data = json.load(json_file)
                    response_data["<step>"] = step # Add step info
          except Exception as e:
               logger.error(f"Error reading movement file {move_file_path}: {e}")
               # Keep response_data as default error state
      else:
           logger.debug(f"Movement file not found for step {step}: {move_file_path}")
           # Keep response_data as default error state ({'<step>': -1})

  except json.JSONDecodeError:
       logger.error("Invalid JSON received in update_environment.")
  except Exception as e:
       logger.error(f"Error updating environment: {e}")

  return JsonResponse(response_data)


# --- path_tester_update view remains the same ---
def path_tester_update(request):
  """
  Processing the path and saving it to path_tester_env.json temp storage for
  conducting the path tester.
  """
  manage_py_dir = os.path.dirname(os.path.abspath(__file__)) # views.py location
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # Go up to frontend_server dir
  temp_storage_dir = os.path.join(base_dir, "temp_storage") # Correct path to temp_storage
  os.makedirs(temp_storage_dir, exist_ok=True) # Ensure directory exists
  output_path = os.path.join(temp_storage_dir, "path_tester_env.json")

  try:
      data = json.loads(request.body)
      camera = data.get("camera") # Use get for safety
      if camera is None:
           logger.error("Missing camera data in path_tester_update.")
           return HttpResponse("Error: Missing camera data.", status=400)

      with open(output_path, "w") as outfile:
          json.dump(camera, outfile, indent=2)

      return HttpResponse("received")
  except json.JSONDecodeError:
       logger.error("Invalid JSON received in path_tester_update.")
       return HttpResponse("Error: Invalid JSON.", status=400)
  except Exception as e:
       logger.error(f"Error in path_tester_update: {e}")
       return HttpResponse("Error processing request.", status=500)

