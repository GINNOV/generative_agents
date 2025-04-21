import google.generativeai as genai
import google.api_core
import os
import re # Added for parsing retry delay in gpt_structure

# SELECT YOUR BACKEND KEY TYPE HERE:
key_type = 'llama' # Options: 'openai', 'azure', 'llama', 'gemini'

# --- VALIDATE KEY TYPE ---
VALID_KEY_TYPES = ['openai', 'azure', 'llama', 'gemini']
assert key_type in VALID_KEY_TYPES, f"ERROR: wrong key type '{key_type}', the key type should select from {VALID_KEY_TYPES}. "

# --- DEFINE PLACEHOLDERS ---
openai_api_key_val = "<Your OpenAI API Key>"
openai_key_owner = "<Name>" # Optional: For reference
azure_api_key_val = "<Your Azure API Key>"
azure_api_base_val = "<Your Azure API Base (e.g., https://YOUR_RESOURCE_NAME.openai.azure.com/)>"
azure_chat_deployment_name_val = "gpt-35-turbo"
azure_completion_deployment_name_val = "text-davinci-003" 
llama_api_base_val = "<Llama API URL (e.g., http://localhost:11434/v1)>" # Ensure OpenAI compatible endpoint /v1
llama_chat_model_name_val = "llama3:latest"
google_api_key_val = "<Your Google API Key (from Google AI Studio or Cloud)>" #use an environment variable to set this

# --- DEFAULT MODEL NAMES ---

# Chat Models (Used by llm_request, ChatGPT_request, etc.)
DEFAULT_OPENAI_CHAT_MODEL = "gpt-3.5-turbo"
DEFAULT_AZURE_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", azure_chat_deployment_name_val)
DEFAULT_LLAMA_CHAT_MODEL = llama_chat_model_name_val 
DEFAULT_GEMINI_CHAT_MODEL = "gemini-1.5-flash-latest"

# Completion Models (Used by legacy functions like run_gpt_prompt_wake_up_hour)
DEFAULT_OPENAI_COMPLETION_MODEL = "gpt-3.5-turbo-instruct"
DEFAULT_AZURE_COMPLETION_MODEL = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME", azure_completion_deployment_name_val)
# Gemini doesn't have a direct completion equivalent, map to a chat model
DEFAULT_GEMINI_COMPLETION_MODEL = "gemini-1.5-flash-latest"

# Embedding Models
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_AZURE_EMBEDDING_MODEL = "text-embedding-ada-002" # Often the same deployment name
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/embedding-001"
DEFAULT_LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Local embedding model for Llama all-MiniLM-L6-v2

# --- DEFAULT API CONFIG VALUES (Will be populated based on key_type) ---
openai_api_key = None
openai_api_base = None
openai_api_type = None
openai_api_version = None
openai_completion_api_key = None # Specific for Azure Completion legacy? Usually same as main key
openai_completion_api_base = None # Specific for Azure Completion legacy? Usually same as main base
google_api_key = None # Specific variable for Gemini Key (populated below)

# --- CONFIGURE BASED ON key_type ---

# OpenAI
if key_type == 'openai':
    openai_api_key = os.getenv("OPENAI_API_KEY", openai_api_key_val)
    openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_api_type = 'openai'
    if openai_api_key == openai_api_key_val:
         print("WARNING: Using placeholder OpenAI API Key. Set the OPENAI_API_KEY environment variable.")

# Azure OpenAI
if key_type == 'azure':
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY", azure_api_key_val)
    openai_api_base = os.getenv("AZURE_OPENAI_API_BASE", azure_api_base_val)
    openai_api_type = 'azure'
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", '2023-05-15') # Or your desired Azure API version
    # Azure often uses the same key/base for completion and chat if using newer models
    openai_completion_api_key = openai_api_key # Use main key
    openai_completion_api_base = openai_api_base # Use main base
    if openai_api_key == azure_api_key_val:
         print("WARNING: Using placeholder Azure API Key/Base. Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_BASE environment variables.")

# Llama (via OpenAI-compatible endpoint)
if key_type == 'llama':
    openai_api_key = os.getenv("LLAMA_API_KEY", "none") # Often not needed if endpoint is open
    openai_api_base = os.getenv("LLAMA_API_BASE", llama_api_base_val)
    openai_api_type = 'openai' # Mimics OpenAI API structure
    if openai_api_base == llama_api_base_val:
         print("WARNING: Using placeholder Llama API Base. Set the LLAMA_API_BASE environment variable.")


# Gemini
if key_type == 'gemini':
    # *** FIX: Load from environment OR placeholder, DO NOT overwrite ***
    google_api_key = os.getenv("GOOGLE_API_KEY", google_api_key_val)
    # google_api_key = google_api_key_val # <<< REMOVED THIS BUGGY LINE
    if google_api_key == google_api_key_val:
        print("WARNING: Using placeholder Google API Key. Set the GOOGLE_API_KEY environment variable.")

    # Set other variables to None or 'gemini' to signal downstream code
    openai_api_base = None
    openai_api_type = 'gemini'
    openai_api_version = None
    # IMPORTANT: The actual Gemini client initialization (genai.configure) happens
    # in gpt_structure.py using the result of os.getenv("GOOGLE_API_KEY") primarily.
    # The 'google_api_key' variable here is less critical now but fixed for consistency.

# --- PATHS & OTHER CONFIG ---
# (Paths remain the same)
maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

# Verbose
debug = True # Set to False for less console output

# sim fold
sim_fold = None
def set_fold(path):
    global sim_fold
    sim_fold = path

# Pool Caching Configuration
use_embedding_pool = True # Set to False to disable embedding caching
embedding_pool_path = os.path.join(fs_storage, "public", "embedding_pool.json")

use_policy_pool = True # Set to False to disable policy caching
policy_pool_path = os.path.join(fs_storage, "public", "policy_pool")

use_sub_task_pool = True # Set to False to disable sub-task caching
sub_task_pool_path = os.path.join(fs_storage, "public", "sub_task_pool")

# Record
record_tree_flag = True # Set to False to disable record tree generation

# Feature Switches
use_policy = True # Set to False to disable using cached policies/sub-tasks
use_relationship = True # Set to False to disable relationship tracking/updates
