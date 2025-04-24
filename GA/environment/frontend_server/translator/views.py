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
import logging
import datetime
import traceback # For detailed error logging

from django.shortcuts import render, redirect, HttpResponseRedirect, HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # Potentially needed if CSRF middleware is strict, but prefer token handling
from django.views.decorators.http import require_POST # Ensure view only accepts POST

# --- Global Methods Import (Keep existing robust logic) ---
try:
    from ..global_methods import check_if_file_exists, find_filenames
except (ImportError, ValueError):
    import sys
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reverie', 'backend_server'))
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    try:
        from global_methods import check_if_file_exists, find_filenames
    except ImportError:
        print("ERROR: Could not import global_methods from expected locations in views.py.")
        def check_if_file_exists(path): return False
        def find_filenames(path, suffix): return []

# --- Import the compression function ---
# Adjust the path based on where compress_sim_storage.py actually lives relative to views.py
# Assuming compress_sim_storage.py is in GA/reverie/
try:
    # Path: views.py (translator) -> .. (frontend_server) -> .. (environment) -> .. (GA) -> reverie -> compress_sim_storage.py
    compress_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reverie'))
    if compress_script_path not in sys.path:
        sys.path.append(compress_script_path)
    from compress_sim_storage import compress as compress_simulation_script
except ImportError as e:
    print(f"ERROR: Could not import compress function: {e}. Compression will not work.")
    # Define a dummy function so the view doesn't crash immediately
    def compress_simulation_script(sim_code, fin_code, storage_dir_base, compressed_dir_base):
        raise NotImplementedError("Compression script could not be imported.")

from django.contrib.staticfiles.templatetags.staticfiles import static
# from .models import * # models.py is still likely empty

# Configure logging
logger = logging.getLogger(__name__)

# --- landing view remains the same ---
def landing(request):
  context = {}
  template = "landing/landing.html"
  from django.template.loader import get_template
  try:
      get_template(template)
  except Exception as e:
       logger.warning(f"Template not found via loader: {template}. Redirecting to home. Error: {e}")
       return redirect('home')
  return render(request, template, context)

# --- demo view remains the same ---
def demo(request, sim_code, step, play_speed="2"):
  manage_py_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..'))
  storage_base = os.path.join(base_dir, 'compressed_storage')
  move_file = os.path.join(storage_base, sim_code, "master_movement.json")
  meta_file = os.path.join(storage_base, sim_code, "meta.json")
  persona_folder = os.path.join(storage_base, sim_code, "personas")

  if not os.path.exists(move_file) or not os.path.exists(meta_file):
       logger.error(f"Compressed simulation data not found for sim_code: {sim_code} at {storage_base}")
       return HttpResponse(f"Error: Compressed simulation data not found for '{sim_code}'. Check folder name and ensure compression was successful.", status=404)

  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4, "4": 8, "5": 16, "6": 32}
  play_speed = play_speed_opt.get(play_speed, 2)

  meta = dict()
  try:
    with open (meta_file) as json_file: meta = json.load(json_file)
  except Exception as e:
       logger.error(f"Error loading meta file {meta_file}: {e}")
       return HttpResponse(f"Error loading simulation metadata for '{sim_code}'.", status=500)

  sec_per_step = meta.get("sec_per_step", 10)
  start_date_str = meta.get("start_date", "January 1, 2023")
  try:
      base_start_datetime = datetime.datetime.strptime(start_date_str + " 00:00:00", '%B %d, %Y %H:%M:%S')
      current_step_datetime = base_start_datetime + datetime.timedelta(seconds=sec_per_step * step)
      start_datetime_iso = current_step_datetime.strftime("%Y-%m-%dT%H:%M:%S")
  except ValueError as e:
       logger.error(f"Error parsing date '{start_date_str}' from meta file: {e}")
       start_datetime_iso = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

  raw_all_movement = dict()
  try:
    with open(move_file) as json_file: raw_all_movement = json.load(json_file)
  except Exception as e:
       logger.error(f"Error loading movement file {move_file}: {e}")
       return HttpResponse(f"Error loading simulation movement data for '{sim_code}'.", status=500)

  max_step_in_data = 0
  if raw_all_movement:
       digit_keys = [int(k) for k in raw_all_movement.keys() if k.isdigit()]
       if digit_keys: max_step_in_data = max(digit_keys)

  persona_names = []
  persona_names_set = set()
  first_step_key = "0"
  if first_step_key in raw_all_movement and raw_all_movement[first_step_key]:
       for p in list(raw_all_movement[first_step_key].keys()):
            initial = p[0] + p.split(" ")[-1][0] if " " in p else p[:2]
            persona_names.append({"original": p, "underscore": p.replace(" ", "_"), "initial": initial.upper()})
            persona_names_set.add(p)
  elif os.path.exists(persona_folder):
       try:
            for p_folder in os.listdir(persona_folder):
                 if os.path.isdir(os.path.join(persona_folder, p_folder)) and not p_folder.startswith('.'):
                      p = p_folder
                      initial = p[0] + p.split(" ")[-1][0] if " " in p else p[:2]
                      persona_names.append({"original": p, "underscore": p.replace(" ", "_"), "initial": initial.upper()})
                      persona_names_set.add(p)
       except Exception as e: logger.error(f"Error listing persona folders in {persona_folder}: {e}")

  step = min(step, max_step_in_data)

  init_prep = dict()
  start_step_str = str(step)
  if start_step_str in raw_all_movement:
       val = raw_all_movement[start_step_str]
       for p in persona_names_set:
            if p in val and "movement" in val[p] and val[p]["movement"] is not None:
                 init_prep[p] = val[p]
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
                 else:
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
       if p_orig in init_prep and "movement" in init_prep[p_orig] and init_prep[p_orig]["movement"] is not None:
            persona_init_pos[p_under] = init_prep[p_orig]["movement"]
       else:
            persona_init_pos[p_under] = [0, 0]
            logger.warning(f"Initial position missing or invalid for {p_orig} at step {step} after check, using default.")

  all_movement_int_keys = {}
  for int_key in range(max_step_in_data + 1):
      key = str(int_key)
      if key in raw_all_movement: all_movement_int_keys[int_key] = raw_all_movement[key]

  context = {"sim_code": sim_code, "step": step, "max_step": max_step_in_data,
             "persona_names": persona_names, "persona_init_pos": json.dumps(persona_init_pos),
             "all_movement": json.dumps(all_movement_int_keys),
             "start_datetime": start_datetime_iso, "base_start_datetime_iso": base_start_datetime.strftime("%Y-%m-%dT%H:%M:%S"),
             "sec_per_step": sec_per_step, "play_speed": play_speed, "mode": "demo"}
  template = "demo/playback.html"
  return render(request, template, context)

# --- home view remains the same ---
def home(request):
    manage_py_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(manage_py_dir, '..'))

    compressed_storage_dir = os.path.join(base_dir, "compressed_storage")
    available_compressed_sims = []
    if os.path.isdir(compressed_storage_dir):
        try:
            available_compressed_sims = sorted([
                d for d in os.listdir(compressed_storage_dir)
                if os.path.isdir(os.path.join(compressed_storage_dir, d)) and not d.startswith('.')
            ])
        except Exception as e: logger.error(f"Error scanning compressed_storage directory: {e}")
    else: logger.warning(f"Compressed storage directory not found: {compressed_storage_dir}")

    uncompressed_storage_dir = os.path.join(base_dir, "storage")
    available_uncompressed_sims = []
    if os.path.isdir(uncompressed_storage_dir):
        try:
            available_uncompressed_sims = sorted([
                d for d in os.listdir(uncompressed_storage_dir)
                if os.path.isdir(os.path.join(uncompressed_storage_dir, d))
                   and not d.startswith('.') and d != 'public' and not d.startswith('base_')
            ])
        except Exception as e: logger.error(f"Error scanning storage directory: {e}")
    else: logger.warning(f"Storage directory not found: {uncompressed_storage_dir}")

    temp_storage_dir = os.path.join(base_dir, "temp_storage")
    f_curr_sim_code = os.path.join(temp_storage_dir, "curr_sim_code.json")
    live_backend_status = {"running": False, "sim_code": None, "step": None}
    if check_if_file_exists(f_curr_sim_code):
        try:
            with open(f_curr_sim_code) as json_file:
                data = json.load(json_file)
                live_backend_status["sim_code"] = data.get("sim_code")
                live_backend_status["running"] = bool(live_backend_status["sim_code"])
        except Exception as e:
            logger.error(f"Error reading live backend status file {f_curr_sim_code}: {e}")
            live_backend_status["running"] = False
    else: logger.info(f"Live backend status file not found: {f_curr_sim_code}")

    context = {
        "available_compressed_sims": available_compressed_sims,
        "available_uncompressed_sims": available_uncompressed_sims,
        "live_backend": live_backend_status,
        "mode": "home"
    }
    template = "home/home.html"
    return render(request, template, context)

# --- replay view remains the same ---
def replay(request, sim_code, step):
  manage_py_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..'))
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
                      persona_names.append([p, p.replace(" ", "_"), initial.upper()])
                      persona_names_set.add(p)
       except Exception as e: logger.error(f"Error listing persona folders in {persona_folder}: {e}")
  else: logger.warning(f"Persona folder not found for replay: {persona_folder}")

  persona_init_pos = []
  environment_folder = os.path.join(sim_storage_path, "environment")
  file_count = []
  latest_step = 0
  if os.path.isdir(environment_folder):
       try:
            digit_files = [int(f.split(".")[0]) for f in os.listdir(environment_folder)
                           if f.endswith('.json') and f[:-5].isdigit()]
            if digit_files: latest_step = max(digit_files)
       except Exception as e: logger.error(f"Error listing environment files in {environment_folder}: {e}")

  step = min(step, latest_step)
  curr_json_path = os.path.join(environment_folder, f'{step}.json')

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
            for p_name_list in persona_names: persona_init_pos.append([p_name_list[0], 0, 0])
  else:
       logger.warning(f"Initial environment file not found for step {step}: {curr_json_path}")
       for p_name_list in persona_names: persona_init_pos.append([p_name_list[0], 0, 0])

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

  context = {"sim_code": sim_code, "step": step, "max_step": latest_step,
             "persona_names": persona_names, "persona_init_pos": persona_init_pos,
             "start_datetime": start_datetime_iso, "base_start_datetime_iso": base_start_datetime.strftime("%Y-%m-%dT%H:%M:%S"),
             "sec_per_step": sec_per_step, "mode": "replay"}
  template = "home/home.html"
  return render(request, template, context)

# --- replay_persona_state view remains the same ---
def replay_persona_state(request, sim_code, step, persona_name):
  manage_py_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..'))
  storage_base = os.path.join(base_dir, 'storage')
  compressed_storage_base = os.path.join(base_dir, 'compressed_storage')
  sim_storage_path = os.path.join(storage_base, sim_code)
  compressed_storage_path = os.path.join(compressed_storage_base, sim_code)
  persona_name_underscore = persona_name.replace(" ", "_")
  persona_name_orig = persona_name.replace("_", " ")

  memory_base = None
  compressed_memory_path = os.path.join(compressed_storage_path, "personas", persona_name_orig, "bootstrap_memory")
  uncompressed_memory_path = os.path.join(sim_storage_path, "personas", persona_name_orig, "bootstrap_memory")

  if os.path.exists(compressed_memory_path): memory_base = compressed_memory_path
  elif os.path.exists(uncompressed_memory_path): memory_base = uncompressed_memory_path
  else:
       compressed_persona_path = os.path.join(compressed_storage_path, "personas", persona_name_orig)
       uncompressed_persona_path = os.path.join(sim_storage_path, "personas", persona_name_orig)
       if os.path.exists(compressed_persona_path) or os.path.exists(uncompressed_persona_path):
            logger.warning(f"Persona folder exists for {persona_name_orig} but bootstrap_memory subfolder not found.")
            memory_base = compressed_memory_path if os.path.exists(compressed_persona_path) else uncompressed_persona_path
       else:
            logger.error(f"Persona folder not found for {persona_name_orig} in sim {sim_code}")
            return HttpResponse(f"Error: Persona data not found for '{persona_name_orig}' in simulation '{sim_code}'.", status=404)

  scratch, spatial, associative = {}, {}, {}
  if memory_base:
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

  a_mem_event, a_mem_chat, a_mem_thought = [], [], []
  try:
       node_keys = sorted([k for k in associative.keys() if isinstance(k, str)],
                           key=lambda x: int(x.split('_')[-1]) if x.startswith('node_') else float('inf'))
  except (ValueError, TypeError): node_keys = sorted(associative.keys())

  for node_id in reversed(node_keys):
    node_details = associative.get(node_id)
    if not node_details or not isinstance(node_details, dict): continue
    node_type = node_details.get("type")
    if node_type == "event": a_mem_event.append(node_details)
    elif node_type == "chat": a_mem_chat.append(node_details)
    elif node_type == "thought": a_mem_thought.append(node_details)

  context = {"sim_code": sim_code, "step": step, "persona_name": persona_name_orig,
             "persona_name_underscore": persona_name_underscore, "scratch": scratch,
             "spatial": spatial, "a_mem_event": a_mem_event, "a_mem_chat": a_mem_chat,
             "a_mem_thought": a_mem_thought}
  template = "persona_state/persona_state.html"
  persona_state_template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', template)
  if not os.path.exists(persona_state_template_path):
       logger.error(f"Template not found: {persona_state_template_path}")
       return HttpResponse("Error: Persona state template missing.", status=500)
  return render(request, template, context)

# --- path_tester view remains the same ---
def path_tester(request):
  context = {}
  template = "path_tester/path_tester.html"
  path_tester_template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', template)
  if not os.path.exists(path_tester_template_path):
       logger.warning(f"Template not found: {path_tester_template_path}. Rendering basic response.")
       return HttpResponse("Path Tester page not fully configured.")
  return render(request, template, context)

# --- process_environment view remains the same ---
def process_environment(request):
  manage_py_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..'))
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
      os.makedirs(sim_env_path, exist_ok=True)
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
  manage_py_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..'))
  storage_base = os.path.join(base_dir, 'storage')
  response_data = {"<step>": -1}
  try:
      data = json.loads(request.body)
      step = data.get("step")
      sim_code = data.get("sim_code")
      if step is None or sim_code is None:
           logger.error("Missing data in update_environment request.")
           return JsonResponse(response_data)
      move_file_path = os.path.join(storage_base, sim_code, "movement", f"{step}.json")
      if check_if_file_exists(move_file_path):
          try:
               with open(move_file_path) as json_file:
                    response_data = json.load(json_file)
                    response_data["<step>"] = step
          except Exception as e: logger.error(f"Error reading movement file {move_file_path}: {e}")
      else: logger.debug(f"Movement file not found for step {step}: {move_file_path}")
  except json.JSONDecodeError: logger.error("Invalid JSON received in update_environment.")
  except Exception as e: logger.error(f"Error updating environment: {e}")
  return JsonResponse(response_data)

# --- path_tester_update view remains the same ---
def path_tester_update(request):
  manage_py_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.abspath(os.path.join(manage_py_dir, '..'))
  temp_storage_dir = os.path.join(base_dir, "temp_storage")
  os.makedirs(temp_storage_dir, exist_ok=True)
  output_path = os.path.join(temp_storage_dir, "path_tester_env.json")
  try:
      data = json.loads(request.body)
      camera = data.get("camera")
      if camera is None:
           logger.error("Missing camera data in path_tester_update.")
           return HttpResponse("Error: Missing camera data.", status=400)
      with open(output_path, "w") as outfile: json.dump(camera, outfile, indent=2)
      return HttpResponse("received")
  except json.JSONDecodeError:
       logger.error("Invalid JSON received in path_tester_update.")
       return HttpResponse("Error: Invalid JSON.", status=400)
  except Exception as e:
       logger.error(f"Error in path_tester_update: {e}")
       return HttpResponse("Error processing request.", status=500)


# --- NEW VIEW for Compression ---
@require_POST # Only allow POST requests
# @csrf_exempt # Use this decorator ONLY if you are having CSRF issues and understand the risks.
# It's better to configure CSRF properly on the frontend.
def compress_simulation_view(request):
    """
    Handles AJAX requests from the frontend to trigger simulation compression.
    """
    try:
        # Load data from request body
        data = json.loads(request.body)
        sim_code = data.get('sim_code')
        fin_code = data.get('fin_code')

        # --- Basic Input Validation ---
        if not sim_code or not fin_code:
            return JsonResponse({'status': 'error', 'message': 'Missing sim_code or fin_code.'}, status=400)

        # More robust validation (prevent directory traversal, etc.)
        valid_chars = set(string.ascii_letters + string.digits + '_-')
        if not all(c in valid_chars for c in sim_code):
             return JsonResponse({'status': 'error', 'message': f"Invalid characters in sim_code: '{sim_code}'."}, status=400)
        if not all(c in valid_chars for c in fin_code):
             return JsonResponse({'status': 'error', 'message': f"Invalid characters in fin_code: '{fin_code}'."}, status=400)

        # --- Determine Absolute Paths ---
        # Assuming views.py is inside an app folder within frontend_server
        manage_py_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(manage_py_dir, '..')) # frontend_server dir
        storage_dir_base = os.path.join(base_dir, 'storage')
        compressed_dir_base = os.path.join(base_dir, 'compressed_storage')

        # Check if source simulation exists
        source_sim_path = os.path.join(storage_dir_base, sim_code)
        if not os.path.isdir(source_sim_path):
            return JsonResponse({'status': 'error', 'message': f"Source simulation '{sim_code}' not found."}, status=404)

        # Check if destination already exists (optional: prevent overwrite or ask user?)
        dest_sim_path = os.path.join(compressed_dir_base, fin_code)
        if os.path.exists(dest_sim_path):
             # For now, let's allow overwrite, the script might handle deletion anyway
             logger.warning(f"Compressed destination '{fin_code}' already exists. It might be overwritten.")
             # Alternatively, return an error:
             # return JsonResponse({'status': 'error', 'message': f"Compressed name '{fin_code}' already exists."}, status=400)


        # --- Call the Compression Script ---
        logger.info(f"View received request to compress '{sim_code}' into '{fin_code}'")
        try:
            # Pass the absolute base paths to the function
            compress_simulation_script(sim_code, fin_code, storage_dir_base, compressed_dir_base)
            logger.info(f"Successfully initiated compression for '{sim_code}'")
            return JsonResponse({'status': 'success', 'message': f"Compression successful for '{sim_code}' into '{fin_code}'. Refresh page."})

        # --- Handle Specific Errors from the Script ---
        except FileNotFoundError as e:
            logger.error(f"Compression failed for '{sim_code}': File not found - {e}")
            return JsonResponse({'status': 'error', 'message': f"Compression failed: {e}"}, status=404)
        except ValueError as e: # Catch validation errors from compress function
            logger.error(f"Compression failed for '{sim_code}': Invalid input - {e}")
            return JsonResponse({'status': 'error', 'message': f"Compression failed: {e}"}, status=400)
        except NotImplementedError as e: # Catch if the import failed
             logger.error(f"Compression feature unavailable: {e}")
             return JsonResponse({'status': 'error', 'message': 'Compression feature is not available (script import failed).'}, status=501)
        except Exception as e:
            logger.error(f"Compression failed unexpectedly for '{sim_code}': {type(e).__name__} - {e}", exc_info=True)
            # traceback.print_exc() # Log detailed traceback to Django console/logs
            return JsonResponse({'status': 'error', 'message': f"An unexpected error occurred during compression: {type(e).__name__}"}, status=500)

    except json.JSONDecodeError:
        logger.error("Invalid JSON received in compress_simulation_view.")
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON in request body.'}, status=400)
    except Exception as e:
        logger.error(f"Generic error in compress_simulation_view: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'message': 'An internal server error occurred.'}, status=500)
