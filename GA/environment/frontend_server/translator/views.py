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
    # Assumes views.py is in the 'translator' app directory
    from ..global_methods import check_if_file_exists, find_filenames
except (ImportError, ValueError): # Catch ValueError if relative import goes too high
    # Fallback if not found locally (adjust path if necessary)
    # This might be needed if the app structure is different or running manage.py from elsewhere
    import sys
    # Construct path to GA/reverie/backend_server relative to this file
    # views.py -> translator -> frontend_server -> environment -> GA -> reverie -> backend_server
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reverie', 'backend_server'))
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    try:
        # Now try importing assuming it's in the backend_server path
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
  # Get correct template path relative to Django's template loaders
  from django.template.loader import get_template
  try:
      get_template(template) # Check if template exists via Django loaders
  except Exception as e:
       logger.warning(f"Template not found via loader: {template}. Redirecting to home. Error: {e}")
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
       # Handle error: render an error template or redirect
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
      # Calculate the base start time for step 0
      base_start_datetime = datetime.datetime.strptime(start_date_str + " 00:00:00",
                                                  '%B %d, %Y %H:%M:%S')
      # Calculate the datetime for the requested starting step
      current_step_datetime = base_start_datetime + datetime.timedelta(seconds=sec_per_step * step)
      start_datetime_iso = current_step_datetime.strftime("%Y-%m-%dT%H:%M:%S")
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

  # --- Calculate Max Step --- START ---
  max_step_in_data = 0
  if raw_all_movement:
       # Filter out non-digit keys before finding max
       digit_keys = [int(k) for k in raw_all_movement.keys() if k.isdigit()]
       if digit_keys:
            max_step_in_data = max(digit_keys)
  # --- Calculate Max Step --- END ---


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

  # Ensure requested step is valid and not beyond available data
  step = min(step, max_step_in_data)

  # Prepare initial state data for the starting step
  init_prep = dict()
  start_step_str = str(step)
  if start_step_str in raw_all_movement:
       val = raw_all_movement[start_step_str]
       for p in persona_names_set:
            # Check if persona exists in this step's data and has movement info
            if p in val and "movement" in val[p] and val[p]["movement"] is not None:
                 init_prep[p] = val[p]
            # If not in this step, try to find the *last known state* from previous steps
            elif p not in init_prep:
                 last_known_state = None
                 for prev_step in range(step - 1, -1, -1):
                      prev_step_str = str(prev_step)
                      if prev_step_str in raw_all_movement and p in raw_all_movement[prev_step_str]:
                           last_known_state = raw_all_movement[prev_step_str][p]
                           break
                 if last_known_state and "movement" in last_known_state and last_known_state["movement"] is not None:
                      init_prep[p] = last_known_state
                      logger.debug(f"Using state from step {prev_step} for {p} at step {step}")
                 else: # Provide default only if no previous state found
                      init_prep[p] = {"movement": [0,0], "pronunciatio": "❓", "description": "Initial state", "chat": None}
                      logger.warning(f"No valid state found for {p} up to step {step}. Using default.")

  else:
       logger.warning(f"Movement data for exact start step {step} not found. Using defaults.")
       for p in persona_names_set:
            init_prep[p] = {"movement": [0,0], "pronunciatio": "❓", "description": "Initial state", "chat": None}

  persona_init_pos = dict()
  for p_name_obj in persona_names:
       p_orig = p_name_obj["original"]
       p_under = p_name_obj["underscore"]
       # Use the prepared initial state
       if p_orig in init_prep and "movement" in init_prep[p_orig] and init_prep[p_orig]["movement"] is not None:
            persona_init_pos[p_under] = init_prep[p_orig]["movement"]
       else:
            persona_init_pos[p_under] = [0, 0]
            logger.warning(f"Initial position missing or invalid for {p_orig} at step {step} after check, using default.")

  # Load all movement data into a dictionary keyed by integer step
  # This ensures all data is available for the slider
  all_movement_int_keys = {}
  for int_key in range(max_step_in_data + 1):
      key = str(int_key)
      if key in raw_all_movement:
          all_movement_int_keys[int_key] = raw_all_movement[key]


  context = {"sim_code": sim_code,
             "step": step, # Current starting step
             "max_step": max_step_in_data, # Pass max_step to template
             "persona_names": persona_names,
             "persona_init_pos": json.dumps(persona_init_pos),
             "all_movement": json.dumps(all_movement_int_keys), # Send all data keyed by integer
             "start_datetime": start_datetime_iso, # ISO format for initial time display
             "base_start_datetime_iso": base_start_datetime.strftime("%Y-%m-%dT%H:%M:%S"), # Pass base time for JS calculations
             "sec_per_step": sec_per_step,
             "play_speed": play_speed,
             "mode": "demo"}
  template = "demo/demo.html"

  return render(request, template, context)

# --- home view remains the same ---
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
    temp_storage_dir = os.path.join(base_dir, "temp_storage") # Path relative to frontend_server dir
    f_curr_sim_code = os.path.join(temp_storage_dir, "curr_sim_code.json")
    live_backend_status = {
        "running": False,
        "sim_code": None,
        "step": None # Step might be harder to get reliably without backend running
    }
    # Use the global_methods helper function imported earlier
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
      base_start_datetime = datetime.datetime.strptime(start_date_str + " 00:00:00", '%B %d, %Y %H:%M:%S')
      current_step_datetime = base_start_datetime + datetime.timedelta(seconds=sec_per_step * step)
      start_datetime_iso = current_step_datetime.strftime("%Y-%m-%dT%H:%M:%S")
  except ValueError as e:
      logger.error(f"Error parsing date '{start_date_str}' from meta file for replay: {e}")
      start_datetime_iso = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

  # Need to load movement data for replay template if it differs from demo
  # For now, assume home.html can adapt or a dedicated replay.html exists

  context = {"sim_code": sim_code,
             "step": step,
             "max_step": latest_step, # Pass max step for replay too
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos, # Send list of lists [name, x, y]
             "start_datetime": start_datetime_iso, # Add time info
             "base_start_datetime_iso": base_start_datetime.strftime("%Y-%m-%dT%H:%M:%S"), # Pass base time
             "sec_per_step": sec_per_step,        # Add step duration
             "mode": "replay"} # Mode indicates it uses full storage
  template = "home/home.html" # Re-use home.html for replay UI? Needs verification.
  # Ensure home.html template can handle the context for 'replay' mode
  return render(request, template, context)


# --- replay_persona_state view remains the same (with added robustness) ---
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
       # Check if persona folder exists even if bootstrap_memory doesn't
       compressed_persona_path = os.path.join(compressed_storage_path, "personas", persona_name_orig)
       uncompressed_persona_path = os.path.join(sim_storage_path, "personas", persona_name_orig)
       if os.path.exists(compressed_persona_path) or os.path.exists(uncompressed_persona_path):
            logger.warning(f"Persona folder exists for {persona_name_orig} but bootstrap_memory subfolder not found.")
            # Allow rendering with potentially empty memory data
            memory_base = compressed_memory_path if os.path.exists(compressed_persona_path) else uncompressed_persona_path
       else:
            logger.error(f"Persona folder not found for {persona_name_orig} in sim {sim_code}")
            return HttpResponse(f"Error: Persona data not found for '{persona_name_orig}' in simulation '{sim_code}'.", status=404)


  # Load memory files with error handling
  scratch = {}
  spatial = {}
  associative = {}
  if memory_base: # Proceed only if a base path was determined
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
       # Ensure keys are strings before splitting
       node_keys = sorted([k for k in associative.keys() if isinstance(k, str)],
                           key=lambda x: int(x.split('_')[-1]) if x.startswith('node_') else float('inf'))
  except (ValueError, TypeError):
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
  persona_state_template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', template)
  if not os.path.exists(persona_state_template_path):
       logger.error(f"Template not found: {persona_state_template_path}")
       return HttpResponse("Error: Persona state template missing.", status=500)
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

