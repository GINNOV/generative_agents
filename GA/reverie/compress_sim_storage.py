"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modifications by: Gemini

File: compress_sim_storage.py
Description: Compresses a simulation's output storage for replay demos
             by delta-compressing movement data and copying essential files.
             Adapted to be callable as a function and raise exceptions.
"""
import shutil
import json
import argparse
import os
import sys
import logging
import string

# Configure logger for this module
logger = logging.getLogger(__name__)

# Assuming global_methods.py might be needed.
# The import logic here tries to be flexible, but might need adjustment
# based on exactly where compress() is called from within Django.
# It's often safer to pass necessary helper functions or paths as arguments.
try:
    # Try relative import first (if compress is called from a known structure)
    from .global_methods import find_filenames, create_folder_if_not_there
except ImportError:
    try:
        # Try importing assuming it's at the same level or in backend_server
        # This path might be fragile depending on Django's execution context
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend_server'))
        from global_methods import find_filenames, create_folder_if_not_there
    except ImportError:
        logger.error("Could not import global_methods. Ensure it's accessible or pass helpers.")
        # Define dummy functions if absolutely necessary, but it's better to fix the import
        def find_filenames(path, suffix): return []
        def create_folder_if_not_there(path): os.makedirs(path, exist_ok=True)
        # raise ImportError("Could not find global_methods") # Or raise error


def compress(sim_code, fin_code, storage_dir_base, compressed_dir_base):
    """
    Compresses simulation data from sim_code directory into fin_code directory.
    Designed to be called from another script (like a Django view).

    Args:
        sim_code (str): The name of the simulation run folder in storage/.
        fin_code (str): The name of the output folder in compressed_storage/.
        storage_dir_base (str): Absolute path to the base 'storage' directory.
        compressed_dir_base (str): Absolute path to the base 'compressed_storage' directory.

    Raises:
        FileNotFoundError: If essential source directories or files are missing.
        ValueError: If input codes are invalid or potentially unsafe.
        Exception: For other errors during compression (IOError, JSON errors, etc.).
    """
    logger.info(f"Starting compression for sim '{sim_code}' into '{fin_code}'")

    # --- Input Validation (Basic) ---
    # Prevent directory traversal and limit characters
    valid_chars = set(string.ascii_letters + string.digits + '_-')
    if not sim_code or not all(c in valid_chars for c in sim_code):
        raise ValueError(f"Invalid sim_code: '{sim_code}'. Use letters, digits, underscore, hyphen.")
    if not fin_code or not all(c in valid_chars for c in fin_code):
        raise ValueError(f"Invalid fin_code: '{fin_code}'. Use letters, digits, underscore, hyphen.")

    # --- Construct Paths ---
    sim_storage = os.path.join(storage_dir_base, sim_code)
    compressed_storage = os.path.join(compressed_dir_base, fin_code)

    persona_folder = os.path.join(sim_storage, "personas")
    move_folder = os.path.join(sim_storage, "movement")
    # Adjust meta file path assumption if needed
    meta_file_reverie = os.path.join(sim_storage, "reverie", "meta.json") # Original path check
    meta_file_root = os.path.join(sim_storage, "meta.json") # Check root of sim_code folder too

    logger.debug(f"Source storage path: {sim_storage}")
    logger.debug(f"Target compressed storage path: {compressed_storage}")

    # --- Check Source Existence ---
    if not os.path.isdir(sim_storage):
        raise FileNotFoundError(f"Source simulation directory not found: {sim_storage}")
    if not os.path.isdir(move_folder):
        raise FileNotFoundError(f"Source movement directory not found: {move_folder}")

    # Determine meta file path
    meta_file = None
    if os.path.isfile(meta_file_reverie):
         meta_file = meta_file_reverie
    elif os.path.isfile(meta_file_root):
         meta_file = meta_file_root

    if meta_file is None:
         # Allow compression without meta? Maybe not ideal. Raise error for now.
         raise FileNotFoundError(f"Meta file not found at expected locations: {meta_file_reverie} or {meta_file_root}")
    else:
         logger.info(f"Using meta file: {meta_file}")

    # Check personas folder (optional?)
    personas_exist = os.path.isdir(persona_folder)
    if not personas_exist:
        logger.warning(f"Source personas directory not found: {persona_folder}. Personas will not be copied.")


    # --- Main Compression Logic (wrapped in try/except) ---
    try:
        # --- Get Persona Names (Only if folder exists) ---
        persona_names = []
        if personas_exist:
            try:
                 for item in os.listdir(persona_folder):
                     item_path = os.path.join(persona_folder, item)
                     # Check if it's a directory and not hidden
                     if os.path.isdir(item_path) and not item.startswith('.'):
                         persona_names.append(item)
            except OSError as e:
                 logger.error(f"Error listing personas directory {persona_folder}: {e}")
                 # Decide how critical this is. Maybe continue without persona names?
                 personas_exist = False # Treat as missing if listing fails

            if not persona_names:
                 logger.warning(f"No persona folders found in {persona_folder}. Proceeding without persona-specific data.")
            else:
                 logger.info(f"Found personas: {persona_names}")
        else:
             logger.info("Proceeding without persona data (folder was missing).")


        # --- Find Movement Files ---
        try:
             movement_files = [f for f in os.listdir(move_folder)
                               if f.endswith('.json') and f[:-5].isdigit()]
        except OSError as e:
             logger.error(f"Error listing movement files in {move_folder}: {e}")
             raise # Movement files are essential

        if not movement_files:
             raise FileNotFoundError(f"No valid movement json files found in {move_folder}")

        max_move_count = max([int(f.split(".")[0]) for f in movement_files])
        logger.info(f"Found movement steps up to: {max_move_count}")

        # --- Process Movement Data (Delta Compression) ---
        persona_last_move = dict() # Stores the last recorded state for each persona
        master_move = dict()       # Stores the delta-compressed states
        logger.info("Processing movement data...")
        for i in range(max_move_count + 1):
            master_move[i] = dict() # Initialize step entry
            move_file_path = os.path.join(move_folder, f"{str(i)}.json")
            try:
                with open(move_file_path) as json_file:
                    try:
                        i_move_dict_full = json.load(json_file)
                        # Check if the structure is as expected
                        if "persona" not in i_move_dict_full or not isinstance(i_move_dict_full["persona"], dict):
                            logger.warning(f"'persona' key missing or not a dict in {move_file_path}. Skipping step {i}.")
                            continue
                        i_move_dict = i_move_dict_full["persona"]
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode JSON from {move_file_path}. Skipping step {i}.")
                        continue

                    # Iterate through personas found in *this step's* file
                    for p in i_move_dict.keys():
                        # If we are tracking specific personas, ensure this one is tracked
                        if persona_names and p not in persona_names:
                             continue # Skip if persona not in the list from persona_folder

                        current_state = i_move_dict[p]
                        if not isinstance(current_state, dict):
                            logger.warning(f"Persona data for '{p}' in step {i} is not a dictionary. Skipping.")
                            continue

                        last_state = persona_last_move.get(p)

                        store_state = False
                        if i == 0 or last_state is None:
                            store_state = True
                        # Check if any relevant field has changed
                        elif (current_state.get("movement") != last_state.get("movement")
                              or current_state.get("pronunciatio") != last_state.get("pronunciatio")
                              or current_state.get("description") != last_state.get("description")
                              or current_state.get("chat") != last_state.get("chat")):
                            store_state = True

                        if store_state:
                            # Create a clean state dictionary with only the required keys
                            state_to_store = {
                                "movement": current_state.get("movement"),
                                "pronunciatio": current_state.get("pronunciatio"),
                                "description": current_state.get("description"),
                                "chat": current_state.get("chat")
                            }
                            persona_last_move[p] = state_to_store
                            master_move[i][p] = state_to_store

            except FileNotFoundError:
                logger.warning(f"Movement file not found for step {i}: {move_file_path}. Skipping.")
                if i in master_move: master_move.pop(i) # Remove empty step entry
                continue
            except KeyError as e:
                 logger.warning(f"Missing expected key '{e}' in step {i} for persona '{p}' in {move_file_path}. Skipping persona for this step.")
                 continue
            except Exception as e:
                 logger.error(f"ERROR processing step {i} for file {move_file_path}: {type(e).__name__} {e}", exc_info=True)
                 continue # Try to continue with other steps

        # --- Save Compressed Data ---
        logger.info("Saving compressed data...")
        # Use the helper function if imported successfully, otherwise use os.makedirs
        create_folder_if_not_there(compressed_storage)

        master_movement_path = os.path.join(compressed_storage, "master_movement.json")
        try:
            with open(master_movement_path, "w") as outfile:
                json.dump(master_move, outfile, indent=2)
            logger.info(f"Saved compressed movement data to: {master_movement_path}")
        except TypeError as e:
             logger.error(f"Data in master_move not JSON serializable: {e}")
             raise # Re-raise as it indicates a fundamental issue
        except IOError as e:
             logger.error(f"Could not write master movement file: {e}")
             raise # Re-raise IO error

        # Copy meta file
        if meta_file: # Check if meta_file was found
            compressed_meta_path = os.path.join(compressed_storage, "meta.json")
            try:
                shutil.copyfile(meta_file, compressed_meta_path)
                logger.info(f"Copied meta file to: {compressed_meta_path}")
            except FileNotFoundError:
                 logger.error(f"Meta file source disappeared during copy: {meta_file}")
                 # Decide if this is critical - maybe raise error?
            except Exception as e:
                logger.error(f"Could not copy meta file '{meta_file}' to '{compressed_meta_path}': {e}")
                raise # Re-raise copy error

        # Copy personas folder (only if it existed initially)
        if personas_exist:
            compressed_personas_path = os.path.join(compressed_storage, "personas")
            try:
                # Remove existing destination first to avoid merging issues
                if os.path.exists(compressed_personas_path):
                     shutil.rmtree(compressed_personas_path)
                shutil.copytree(persona_folder, compressed_personas_path)
                logger.info(f"Copied personas folder to: {compressed_personas_path}")
            except FileNotFoundError:
                 logger.error(f"Personas source folder disappeared during copy: {persona_folder}")
                 # Decide if critical
            except Exception as e:
                logger.error(f"Could not copy personas folder '{persona_folder}' to '{compressed_personas_path}': {e}")
                raise # Re-raise copy error

        logger.info(f"Compression complete for '{sim_code}' into '{fin_code}'.")

    except Exception as e:
        # Log the specific error and re-raise it for the calling function (view) to handle
        logger.error(f"Compression failed for sim '{sim_code}': {type(e).__name__} - {e}", exc_info=True)
        # Clean up partially created compressed folder? Optional.
        # if os.path.exists(compressed_storage):
        #     try: shutil.rmtree(compressed_storage)
        #     except: logger.error(f"Could not clean up failed compression folder: {compressed_storage}")
        raise # Re-raise the exception


# --- Keep the __main__ block for standalone execution/testing ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compresses simulation storage from <sim_code> folder into <fin_code> folder for replay.",
        usage='%(prog)s sim_code fin_code [storage_base] [compressed_base]'
        )
    parser.add_argument("sim_code", type=str,
                        help="The simulation code (folder name in storage/) to compress.")
    parser.add_argument("fin_code", type=str,
                        help="The desired output folder name that will appear in compressed_storage/")
    # Add optional base path arguments for standalone testing
    parser.add_argument("storage_base", type=str, nargs='?', default="../environment/frontend_server/storage",
                        help="Base path to the 'storage' directory (relative to script or absolute).")
    parser.add_argument("compressed_base", type=str, nargs='?', default="../environment/frontend_server/compressed_storage",
                        help="Base path to the 'compressed_storage' directory (relative to script or absolute).")

    if len(sys.argv) < 3:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Resolve base paths relative to the script if they are relative
    script_dir = os.path.dirname(os.path.abspath(__file__))
    storage_base_abs = os.path.abspath(os.path.join(script_dir, args.storage_base))
    compressed_base_abs = os.path.abspath(os.path.join(script_dir, args.compressed_base))

    # Basic logging setup for standalone run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        compress(args.sim_code, args.fin_code, storage_base_abs, compressed_base_abs)
        print(f"\nStandalone compression successful for '{args.sim_code}'.")
    except Exception as e:
        print(f"\nStandalone compression FAILED for '{args.sim_code}': {e}")
        sys.exit(1)
