import os
import utils
# Import the flag directly if needed, or rely on utils.*
# from utils import record_tree_flag
import atexit
import json

# Global dictionary to store recording tree data
record_tree = dict()

def update_record_tree(agent_name, task, parent, is_new, duration, time, chat):
    """
    Updates the global record_tree dictionary with new task information for an agent.

    Args:
        agent_name (str): The name of the agent.
        task (str): Description of the task or activity.
        parent (str): Description of the parent task or context.
        is_new (bool): Flag indicating if this is a newly generated task vs. reused.
        duration (int): Duration of the task in minutes or steps.
        time (datetime): Timestamp associated with the task.
        chat (list): Associated chat conversation, if any.
    """
    # Check the flag defined in utils.py (or imported directly)
    if utils.record_tree_flag:
        # Initialize agent's record list if not present
        if agent_name not in record_tree:
            record_tree[agent_name] = []
        # Append the new record
        record_tree[agent_name].append({
            'task': task,
            "parent": parent,
            "is_new": is_new,
            "duration": duration,
            "time": str(time), # Convert datetime to string for JSON serialization
            "chat": chat
        })

def save_record_tree():
    """
    Saves the recorded task tree for each agent to a JSON file upon script exit.
    Includes a check to ensure utils.sim_fold is set.
    """
    # Check the flag from utils.py
    if utils.record_tree_flag:
        # --- Added Check ---
        # Ensure the simulation folder path (sim_fold) has been set in utils
        if utils.sim_fold:
            # Construct the path to the metrics directory
            save_path = os.path.join(utils.sim_fold, "metrics")

            # --- Added Directory Creation ---
            # Ensure the metrics directory exists before trying to save files into it
            try:
                os.makedirs(save_path, exist_ok=True)
            except OSError as e:
                print(f"Error creating metrics directory {save_path}: {e}")
                return # Exit if directory cannot be created

            # Iterate through the recorded data for each agent
            for name, info in record_tree.items():
                # Construct the full path for the agent's JSON file
                file_path = os.path.join(save_path, f"{name}.json")
                try:
                    # Open the file and dump the agent's recorded info as JSON
                    with open(file_path, 'w') as f:
                        json.dump(info, f, indent=4)
                        print(f"write record_tree to {file_path}")
                except IOError as e:
                    print(f"Error writing record tree file {file_path}: {e}")
                except TypeError as e:
                    print(f"Error serializing record tree data for {name} to JSON: {e}")

        else:
            # Print a warning if sim_fold was not set (e.g., due to early exit)
            print("Warning: utils.sim_fold not set. Skipping save_record_tree.")
        # --- End Added Check ---

# Register the save function to be called automatically when the script exits
atexit.register(save_record_tree)
