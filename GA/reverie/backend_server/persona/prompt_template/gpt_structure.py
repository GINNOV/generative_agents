# /Users/mario/code/GitHub/gem-Generative-Agents/GA/reverie/backend_server/persona/prompt_template/gpt_structure.py

"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modifications for Gemini by: Your Name/AI Assistant
Modifications for Quota Retry & Centralized Models by: Gemini

File: gpt_structure.py
Description: Wrapper functions for calling LLM APIs (OpenAI, Azure, Llama, Gemini).
"""
import json
import random
import openai
import time
import os
import sys
import inspect
import re

# --- Import Default Model Constants ---
from utils import (
    # Core config
    key_type, debug, use_embedding_pool,

    # API Keys/Bases/Types/Versions
    openai_api_key, openai_api_base, # <<< ADDED THESE
    openai_api_type, openai_api_version,
    openai_completion_api_key, openai_completion_api_base,
    google_api_key, google_api_key_val,

    # Default Model Constants
    DEFAULT_OPENAI_CHAT_MODEL, DEFAULT_AZURE_CHAT_MODEL, DEFAULT_LLAMA_CHAT_MODEL, DEFAULT_GEMINI_CHAT_MODEL,
    DEFAULT_OPENAI_COMPLETION_MODEL, DEFAULT_AZURE_COMPLETION_MODEL, DEFAULT_GEMINI_COMPLETION_MODEL,
    DEFAULT_OPENAI_EMBEDDING_MODEL, DEFAULT_AZURE_EMBEDDING_MODEL, DEFAULT_GEMINI_EMBEDDING_MODEL,
    DEFAULT_LOCAL_EMBEDDING_MODEL)

from metrics import metrics
from pool import get_embedding_pool, update_embedding_pool
from sentence_transformers import SentenceTransformer
# --- End Import ---

# --- Configure API Clients ---

# Google Gemini Client Initialization (Moved here for clarity, uses utils.py vars)
if key_type == 'gemini':
    try:
        import google.generativeai as genai
        import google.api_core # For exception handling
        # Configure Gemini API key using the variable loaded in utils.py
        gemini_api_key_to_use = google_api_key # Use the variable processed in utils.py
        if not gemini_api_key_to_use or gemini_api_key_to_use == google_api_key_val: # Check against placeholder val
            raise ValueError("Gemini API Key not configured. Set GOOGLE_API_KEY environment variable or configure in utils.py")
        genai.configure(api_key=gemini_api_key_to_use)
        print("Gemini API configured.")
    except ImportError:
        print("ERROR: 'google-generativeai' package not installed. Please install it to use the 'gemini' key_type.")
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)

# --- Helper Functions ---

def get_caller_function_names():
    """Gets the stack of caller function names."""
    stack = inspect.stack()
    caller_names = [frame.function for frame in stack][2:]
    return '.'.join(reversed(caller_names))

def temp_sleep(seconds=0.1):
    """Sleep for a specified duration."""
    if seconds <= 0: return
    time.sleep(seconds)

# --- Constants for Quota Retry ---
MAX_QUOTA_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 20 # Fallback delay

def _handle_quota_exception(e, attempt, max_retries, model_name):
    """Handles the ResourceExhausted exception with retry logic."""
    print(f"\n--- Gemini API Quota Limit Hit ({model_name} - Attempt {attempt + 1}/{max_retries}) ---")
    print(f"Error details: {e}")

    if attempt < max_retries - 1:
        suggested_delay = BASE_RETRY_DELAY_SECONDS
        try: # Attempt to parse suggested delay from error metadata/details
            # Check metadata (common in google-api-core >= 1.31.5)
            if hasattr(e, 'metadata') and e.metadata:
                 for item in e.metadata:
                      # Example parsing, adjust key/value format as needed
                      if isinstance(item, tuple) and len(item) == 2:
                           key, value = item
                           if key == 'retry-delay' and isinstance(value, str): # Check common metadata keys
                                if 's' in value: # Simple seconds parsing
                                     suggested_delay = max(BASE_RETRY_DELAY_SECONDS, int(value.split('s')[0]) + 1)
                                elif 'm' in value:
                                     suggested_delay = max(BASE_RETRY_DELAY_SECONDS, int(value.split('m')[0]) * 60 + 1)
                                break # Found delay
            # Fallback: Check details attribute (less common structure)
            elif hasattr(e, 'details') and e.details:
                 details_str = str(e.details()) # Call details() if it's a method
                 if "retry_delay" in details_str:
                      match = re.search(r"seconds:\s*(\d+)", details_str)
                      if match:
                            suggested_delay = max(BASE_RETRY_DELAY_SECONDS, int(match.group(1)) + 1)
        except Exception as parse_err:
            print(f"Could not parse suggested retry delay from error: {parse_err}")

        print(f"Waiting for {suggested_delay} seconds before retrying...")
        time.sleep(suggested_delay)
        return True # Indicate that a retry should occur
    else:
        # --- Max retries reached ---
        print("\n--- Gemini API Quota Exceeded After Retries ---")
        print("Maximum retry attempts reached due to persistent quota limits.")
        print("Please check your usage, billing details, or increase delays between requests.")
        print("Stopping the simulation.")
        print("More info: https://ai.google.dev/gemini-api/docs/rate-limits")
        sys.exit(1) # Exit gracefully

# --- Core API Request Functions ---

def llm_request(prompt,
                model_name="default", # Specify model or use default from utils.py
                temperature=0.7,
                max_tokens=1024,
                top_p=1.0,
                stop_sequences=None,
                time_sleep_second=0.1):
    """
    Handles requests to various LLM backends based on global `key_type`.
    Includes retry logic for Gemini quota errors & immediate exit for other Gemini API errors.
    """
    temp_sleep(time_sleep_second)
    start_time = time.time()
    message = ""
    total_token = 0

    # --- Select Model Based on Backend if Default ---
    effective_model = model_name
    if effective_model == "default":
        # *** USE CENTRALIZED CONSTANTS ***
        if key_type == 'gemini':
            effective_model = DEFAULT_GEMINI_CHAT_MODEL
        elif key_type == 'openai':
            effective_model = DEFAULT_OPENAI_CHAT_MODEL
        elif key_type == 'azure':
            effective_model = DEFAULT_AZURE_CHAT_MODEL
        elif key_type == 'llama':
            effective_model = DEFAULT_LLAMA_CHAT_MODEL
        else:
             raise ValueError(f"Unknown key_type '{key_type}' for default model selection")
        # *** END USE CENTRALIZED CONSTANTS ***

    # --- GEMINI ---
    if key_type == 'gemini':
        if 'google.generativeai' not in sys.modules:
             raise RuntimeError("Gemini libraries not loaded. Cannot make Gemini API call.")

        print(f"--- Making Gemini Request ---")
        print(f"  Model: {effective_model}")
        print(f"  Prompt: {prompt[:100]}...")

        try:
            gemini_model = genai.GenerativeModel(effective_model)
            generation_config = genai.types.GenerationConfig(
                temperature=temperature, max_output_tokens=max_tokens,
                top_p=top_p, stop_sequences=stop_sequences
            )
            safety_settings = [ # Adjust as needed
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            for attempt in range(MAX_QUOTA_RETRIES):
                try:
                    response = gemini_model.generate_content(
                        prompt, generation_config=generation_config, safety_settings=safety_settings
                    )
                    if response.parts: message = response.text
                    else:
                        message = f"ERROR: Gemini response blocked or empty. Reason: {response.prompt_feedback}"
                        print(message)
                    total_token = 0
                    break # Exit loop on success or block

                except google.api_core.exceptions.ResourceExhausted as e:
                    should_retry = _handle_quota_exception(e, attempt, MAX_QUOTA_RETRIES, effective_model)
                    if not should_retry:
                         message = f"ERROR: API Call Failed (Quota Exceeded after retries)"
                         total_token = 0
                         break

                except Exception as e: # Catch other API errors (like invalid key)
                    error_message = f"LLM API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
                    print(error_message)
                    metrics.fail_record(error_message)
                    # *** EXIT ON OTHER ERRORS ***
                    print("Stopping the simulation due to unrecoverable API error.")
                    sys.exit(1)
                    # *** END EXIT ON OTHER ERRORS ***
                    # message = f"ERROR: API Call Failed ({type(e).__name__})" # Unreachable due to exit
                    # total_token = 0 # Unreachable
                    # break # Unreachable

        except Exception as model_init_e: # Catch potential errors during model initialization
            error_message = f"LLM Init ERROR ({key_type}, Model: {effective_model}): {type(model_init_e).__name__} - {model_init_e}"
            print(error_message)
            metrics.fail_record(error_message)
            print("Stopping the simulation due to model initialization error.")
            sys.exit(1)


    # --- AZURE / LLAMA / OPENAI ---
    else:
        try:
            if key_type == 'azure':
                print(f"--- Making Azure Request ---")
                print(f"  Engine: {effective_model}") # Azure uses 'engine'
                print(f"  Prompt: {prompt[:100]}...")
                completion = openai.ChatCompletion.create(
                    api_type=openai_api_type, api_version=openai_api_version,
                    api_base=openai_api_base, api_key=openai_api_key,
                    engine=effective_model, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop_sequences
                )
                message = completion["choices"][0]["message"]["content"]
                total_token = completion['usage']['total_tokens']

            elif key_type == 'llama':
                print(f"--- Making Llama Request ---")
                print(f"  Model: {effective_model}")
                print(f"  Prompt: {prompt[:100]}...")
                completion = openai.ChatCompletion.create(
                    api_base=openai_api_base, api_key=openai_api_key,
                    model=effective_model, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop_sequences
                )
                message = completion["choices"][-1]["message"]["content"]
                total_token = completion.get('usage', {}).get('total_tokens', 0)

            elif key_type == 'openai':
                print(f"--- Making OpenAI Request ---")
                print(f"  Model: {effective_model}")
                print(f"  Prompt: {prompt[:100]}...")
                completion = openai.ChatCompletion.create(
                    api_key=openai_api_key, api_base=openai_api_base,
                    model=effective_model, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop_sequences
                )
                message = completion["choices"][0]["message"]["content"]
                total_token = completion['usage']['total_tokens']

            else: # Should not happen due to check earlier, but for safety
                raise ValueError(f"Unsupported key_type for LLM request: {key_type}")

        except Exception as e:
            error_message = f"LLM API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
            print(error_message)
            metrics.fail_record(error_message)
            message = f"ERROR: API Call Failed ({type(e).__name__})"
            total_token = 0
            # Decide if non-Gemini errors should also exit immediately
            # print("Stopping the simulation due to unrecoverable API error.")
            # sys.exit(1)

    # --- Metrics Recording ---
    function_name = get_caller_function_names()
    time_use = time.time() - start_time
    if not message.startswith("ERROR:"):
        metrics.call_record(function_name, effective_model, total_token, time_use)

    return message


# ============================================================================
# #####################[SECTION 1: CHAT MODEL WRAPPERS] ######################
# ============================================================================

def ChatGPT_single_request(prompt, time_sleep_second=0.1):
    return llm_request(prompt, model_name="default", time_sleep_second=time_sleep_second)

def GPT4_request(prompt):
    """Requests GPT-4 or equivalent high-end model."""
    model_to_use = "gpt-4" # Default OpenAI GPT-4
    if key_type == 'gemini':
        # Map to a capable Gemini model (e.g., Pro)
        # *** USE CENTRALIZED CONSTANT (using CHAT model here) ***
        # Consider adding a separate constant like DEFAULT_GEMINI_ADVANCED_CHAT_MODEL if needed
        model_to_use = DEFAULT_GEMINI_CHAT_MODEL # Or "gemini-1.5-pro-latest" if specifically desired for GPT-4 mapping
        print(f"Mapping GPT-4 request to Gemini model: {model_to_use}")
        # *** END USE CENTRALIZED CONSTANT ***
    elif key_type == 'azure':
        # Use the chat deployment name, assuming it's mapped to GPT-4 on Azure portal
        model_to_use = DEFAULT_AZURE_CHAT_MODEL
        print(f"Mapping GPT-4 request to Azure deployment: {model_to_use}")

    try:
        response_message = llm_request(prompt, model_name=model_to_use, temperature=0.5)
        if response_message.startswith("ERROR:"):
             return "LLM ERROR"
        return response_message
    except Exception as e:
        metrics.fail_record(f"GPT4_request wrapper error: {e}")
        print(f"GPT4_request ERROR: {e}")
        return "LLM ERROR"

def ChatGPT_request(prompt):
    """ Requests the default chat model (e.g., GPT-3.5-turbo or Gemini-Flash). """
    try:
        response_message = llm_request(prompt, model_name="default", time_sleep_second=0)
        if response_message.startswith("ERROR:"):
             return "LLM ERROR"
        return response_message
    except Exception as e:
        metrics.fail_record(f"ChatGPT_request wrapper error: {e}")
        print(f"ChatGPT_request ERROR: {e}")
        return "LLM ERROR"


# --- Safe Generation Wrappers (Using llm_request indirectly) ---
def LLM_safe_generate_response(request_func, prompt, example_output, special_instruction,
                               repeat=3, fail_safe_response="error", func_validate=None,
                               func_clean_up=None, verbose=False):
    """Generic safe generation function using a provided request function."""
    # (Logic remains the same, relies on the underlying request_func)
    full_prompt = '"""\n' + prompt + '\n"""\n'
    full_prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    full_prompt += "Example output json:\n"
    full_prompt += json.dumps({"output": example_output})

    if verbose: print("----------- LLM PROMPT -----------\n", full_prompt, "\n----------------------------------")

    for i in range(repeat):
        raw_response = request_func(full_prompt)
        if raw_response == "LLM ERROR":
             print(f"Attempt {i+1}/{repeat}: LLM request failed.")
             continue
        if verbose: print(f"--- Attempt {i+1}/{repeat} Raw Response ---\n", raw_response, "\n-------------------------------")

        try:
            start_index = raw_response.find('{')
            end_index = raw_response.rfind('}') + 1
            if start_index != -1 and end_index != -1:
                 json_str = raw_response[start_index:end_index]
                 parsed_json = json.loads(json_str)
                 if "output" in parsed_json: curr_gpt_response = parsed_json["output"]
                 else:
                      print(f"Attempt {i+1}/{repeat}: 'output' key missing in parsed JSON.")
                      if verbose: print(f"Parsed JSON: {json_str}")
                      continue
            else:
                 print(f"Attempt {i+1}/{repeat}: Could not find valid JSON object in response.")
                 if verbose: print(f"Response was: {raw_response}")
                 continue

            if func_validate and func_clean_up:
                if func_validate(curr_gpt_response, prompt=full_prompt):
                    return func_clean_up(curr_gpt_response, prompt=full_prompt)
                elif verbose: print(f"Attempt {i+1}/{repeat}: Validation failed for output: {curr_gpt_response}")
            else:
                print("Warning: No validation or cleanup function provided.")
                return curr_gpt_response

        except json.JSONDecodeError as json_e:
            metrics.fail_record(f"JSON Parsing Failed: {json_e} in response: {raw_response[:500]}")
            print(f"Attempt {i+1}/{repeat}: JSON Parsing Error - {json_e}")
            if verbose: print(f"Raw response causing error: {raw_response}")
        except Exception as e:
            metrics.fail_record(f"Safe Generate Processing Error: {type(e).__name__} - {e}")
            print(f"Attempt {i+1}/{repeat}: Unexpected Error - {type(e).__name__}: {e}")

    print(f"LLM_safe_generate_response failed after {repeat} attempts.")
    return fail_safe_response


def GPT4_safe_generate_response(*args, **kwargs):
    return LLM_safe_generate_response(GPT4_request, *args, **kwargs)

def ChatGPT_safe_generate_response(*args, **kwargs):
    return LLM_safe_generate_response(ChatGPT_request, *args, **kwargs)


# --- Legacy Safe Generate (Not Recommended for JSON) ---
def ChatGPT_safe_generate_response_OLD(prompt, repeat=3, fail_safe_response="error",
                                       func_validate=None, func_clean_up=None, verbose=False):
    """Older safe generation without explicit JSON structure assumption."""
    # (Logic remains the same, relies on ChatGPT_request -> llm_request)
    if verbose: print("----------- LEGACY CHATGPT PROMPT -----------\n", prompt, "\n-------------------------------------------")

    for i in range(repeat):
        curr_gpt_response = ChatGPT_request(prompt)
        if curr_gpt_response == "LLM ERROR":
             print(f"Attempt {i+1}/{repeat}: LLM request failed.")
             continue
        response_to_validate = curr_gpt_response.strip()
        try:
            if func_validate and func_clean_up:
                 if func_validate(response_to_validate, prompt=prompt):
                      return func_clean_up(response_to_validate, prompt=prompt)
                 elif verbose: print(f"---- Legacy repeat count: {i} ----\nValidation Failed Response: {response_to_validate}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            else:
                 print("Warning: No validation or cleanup function provided.")
                 return response_to_validate
        except Exception as e:
            error_info = f"Legacy Safe Generate Processing Error: {type(e).__name__} - {e}"
            metrics.fail_record(error_info)
            print(f"Attempt {i+1}/{repeat}: Error during legacy validation/cleanup - {error_info}")

    print("Legacy FAIL SAFE TRIGGERED")
    return fail_safe_response

# ============================================================================
# ###################[SECTION 2: LEGACY COMPLETION API] ###################
# ============================================================================

def gpt_request_all_version(prompt, gpt_parameter):
    """
    Handles requests for legacy OpenAI Completion models and attempts
    to map to Gemini's generate_content if key_type is 'gemini'.
    Includes retry logic for Gemini quota errors & immediate exit for other Gemini API errors.
    """
    start_time = time.time()
    response_text = ""
    total_token = 0
    # Use model/engine from gpt_parameter, fallback to default *completion* model from utils
    model_name = gpt_parameter.get("engine", gpt_parameter.get("model", None))
    if model_name is None: # Determine default based on key_type if not provided
         if key_type == 'gemini': model_name = DEFAULT_GEMINI_COMPLETION_MODEL
         elif key_type == 'openai': model_name = DEFAULT_OPENAI_COMPLETION_MODEL
         elif key_type == 'azure': model_name = DEFAULT_AZURE_COMPLETION_MODEL
         elif key_type == 'llama': model_name = DEFAULT_LLAMA_CHAT_MODEL
         else: model_name = "text-davinci-003" # Generic fallback?

    # --- GEMINI (Attempted Mapping) ---
    if key_type == 'gemini':
        if 'google.generativeai' not in sys.modules:
             raise RuntimeError("Gemini libraries not loaded.")

        # Use the determined model name (likely DEFAULT_GEMINI_COMPLETION_MODEL)
        mapped_gemini_model = model_name
        print(f"--- Making Gemini Request (Mapped from Legacy Completion) ---")
        print(f"  Legacy Model Param: {gpt_parameter.get('engine', gpt_parameter.get('model', 'N/A'))} (mapped to {mapped_gemini_model})")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Warning: Legacy frequency/presence penalty parameters ignored for Gemini.")

        try:
            gemini_model = genai.GenerativeModel(mapped_gemini_model)
            generation_config = genai.types.GenerationConfig(
                temperature=gpt_parameter.get("temperature", 0.7),
                max_output_tokens=gpt_parameter.get("max_tokens", 1024),
                top_p=gpt_parameter.get("top_p", 1.0),
                stop_sequences=gpt_parameter.get("stop", None)
            )
            safety_settings = [ # Use same safety settings as llm_request
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            for attempt in range(MAX_QUOTA_RETRIES):
                try:
                    response = gemini_model.generate_content(
                        prompt, generation_config=generation_config, safety_settings=safety_settings
                    )
                    if response.parts: response_text = response.text
                    else:
                        response_text = f"ERROR: Gemini response blocked or empty. Reason: {response.prompt_feedback}"
                        print(response_text)
                    total_token = 0
                    break # Exit loop on success or block

                except google.api_core.exceptions.ResourceExhausted as e:
                    should_retry = _handle_quota_exception(e, attempt, MAX_QUOTA_RETRIES, mapped_gemini_model)
                    if not should_retry:
                         response_text = "ERROR: API Call Failed (Quota Exceeded after retries)"
                         total_token = 0
                         break

                except Exception as e: # Catch other API errors
                    error_message = f"Legacy Completion Map to Gemini ERROR ({mapped_gemini_model}): {type(e).__name__} - {e}"
                    print(error_message)
                    metrics.fail_record(error_message)
                    # *** EXIT ON OTHER ERRORS ***
                    print("Stopping the simulation due to unrecoverable API error.")
                    sys.exit(1)
                    # *** END EXIT ON OTHER ERRORS ***
                    # response_text = "ERROR: API Call Failed" # Unreachable
                    # total_token = 0 # Unreachable
                    # break # Unreachable

        except Exception as model_init_e: # Catch potential errors during model initialization
            error_message = f"LLM Init ERROR ({key_type}, Model: {mapped_gemini_model}): {type(model_init_e).__name__} - {model_init_e}"
            print(error_message)
            metrics.fail_record(error_message)
            print("Stopping the simulation due to model initialization error.")
            sys.exit(1)

    # --- LLAMA (Mapped via Chat Completion Endpoint) ---
    elif key_type == 'llama':
        print(f"--- Making Llama Request (Mapped from Legacy Completion) ---")
        print(f"  Model: {model_name}")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Warning: Legacy frequency/presence penalty parameters ignored for Llama.")

        # Use the ChatCompletion endpoint even for legacy requests
        completion = openai.ChatCompletion.create(
            api_base=openai_api_base, # From utils.py
            api_key=openai_api_key,   # From utils.py ("none" usually)
            model=model_name, # Use the model name (e.g., DEFAULT_LLAMA_CHAT_MODEL)
            messages=[{"role": "user", "content": prompt}], # Treat legacy prompt as user message
            temperature=gpt_parameter.get("temperature", 0.7),
            max_tokens=gpt_parameter.get("max_tokens", 1024),
            top_p=gpt_parameter.get("top_p", 1.0),
            # stream=gpt_parameter.get("stream", False), # Stream might work differently
            stop=gpt_parameter.get("stop", None)
        )
        # Llama response structure might vary, adjust if needed
        response_text = completion["choices"][-1]["message"]["content"]
        # Token usage might not be standard or available
        total_token = completion.get('usage', {}).get('total_tokens', 0)

    # --- AZURE / OPENAI (Legacy Completion Endpoint) ---
    else:
        try:
            if gpt_parameter.get('api_type') == 'azure' or key_type == 'azure':
                azure_comp_base = openai_completion_api_base or openai_api_base
                azure_comp_key = openai_completion_api_key or openai_api_key
                if not azure_comp_base or not azure_comp_key:
                     raise ValueError("Azure Completion API base or key not configured.")
                print(f"--- Making Azure Legacy Completion Request ---")
                print(f"  Engine: {model_name}")
                print(f"  Prompt: {prompt[:100]}...")
                response = openai.Completion.create(
                    api_base=azure_comp_base, api_key=azure_comp_key, api_type='azure',
                    api_version=openai_api_version, engine=model_name, prompt=prompt,
                    temperature=gpt_parameter.get("temperature", 0.7), max_tokens=gpt_parameter.get("max_tokens", 1024),
                    top_p=gpt_parameter.get("top_p", 1.0), frequency_penalty=gpt_parameter.get("frequency_penalty", 0),
                    presence_penalty=gpt_parameter.get("presence_penalty", 0), stream=gpt_parameter.get("stream", False),
                    stop=gpt_parameter.get("stop", None)
                )
                response_text = response.choices[0].text
                total_token = response['usage']['total_tokens']
            else: # Assume standard OpenAI
                 print(f"--- Making OpenAI Legacy Completion Request ---")
                 print(f"  Model: {model_name}")
                 print(f"  Prompt: {prompt[:100]}...")
                 response = openai.Completion.create(
                    api_key=openai_api_key, api_base=openai_api_base, model=model_name, prompt=prompt,
                    temperature=gpt_parameter.get("temperature", 0.7), max_tokens=gpt_parameter.get("max_tokens", 1024),
                    top_p=gpt_parameter.get("top_p", 1.0), frequency_penalty=gpt_parameter.get("frequency_penalty", 0),
                    presence_penalty=gpt_parameter.get("presence_penalty", 0), stream=gpt_parameter.get("stream", False),
                    stop=gpt_parameter.get("stop", None)
                 )
                 response_text = response.choices[0].text
                 total_token = response['usage']['total_tokens']

        except Exception as e:
            error_message = f"Legacy Completion API ERROR ({key_type}, Model: {model_name}): {type(e).__name__} - {e}"
            print(error_message)
            metrics.fail_record(error_message)
            response_text = "ERROR: API Call Failed"
            total_token = 0
            # Decide if non-Gemini errors should also exit immediately
            # print("Stopping the simulation due to unrecoverable API error.")
            # sys.exit(1)

    # --- Metrics Recording ---
    function_name = get_caller_function_names()
    time_use = time.time() - start_time
    if not response_text.startswith("ERROR:"):
        metrics.call_record(function_name, model_name, total_token, time_use)

    return response_text


def GPT_request(prompt, gpt_parameter):
    """Legacy wrapper for gpt_request_all_version."""
    temp_sleep()
    try:
        response = gpt_request_all_version(prompt, gpt_parameter)
        # Check for specific error messages returned by gpt_request_all_version
        if response == "ERROR: API Call Failed":
             print("TOKEN LIMIT EXCEEDED or other API Error")
             return "TOKEN LIMIT EXCEEDED"
        elif response == "ERROR: API Call Failed (Quota Exceeded after retries)":
             print("Quota limit exceeded after retries.")
             # For compatibility with existing checks, still return the old message
             return "TOKEN LIMIT EXCEEDED"
        return response
    except Exception as e:
        metrics.fail_record(f"GPT_request wrapper error: {e}")
        print(f"GPT_request ERROR (wrapper): {e}")
        print("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"

# --- Legacy Prompt Generation and Safe Execution ---
def generate_prompt(curr_input, prompt_lib_file):
    """Generates a prompt string from a template file and input(s)."""
    # (Logic remains the same)
    if isinstance(curr_input, str): curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]
    try:
        with open(prompt_lib_file, "r") as f: prompt = f.read()
    except FileNotFoundError:
         print(f"Error: Prompt file not found at {prompt_lib_file}")
         # Consider raising error or returning specific indicator
         # raise FileNotFoundError(f"Prompt file not found: {prompt_lib_file}")
         return "" # Return empty string if file not found
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>", 1)[1]
    return prompt.strip()


def safe_generate_response(prompt, gpt_parameter, repeat=5, fail_safe_response="error",
                           func_validate=None, func_clean_up=None, verbose=False):
    """Safe generation using the legacy GPT_request function."""
    # (Logic remains the same, relies on GPT_request -> gpt_request_all_version)
    if verbose: print("----------- LEGACY SAFE GEN PROMPT -----------\n", prompt, "\n---------------------------------------------")
    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)
        if curr_gpt_response == "TOKEN LIMIT EXCEEDED":
            print(f"Attempt {i+1}/{repeat}: Legacy LLM request failed (TOKEN LIMIT EXCEEDED or Quota).")
            continue
        elif curr_gpt_response == "ERROR: API Call Failed":
             print(f"Attempt {i+1}/{repeat}: Legacy LLM request failed (API Error).")
             continue
        try:
            if func_validate and func_clean_up:
                response_to_validate = curr_gpt_response
                if func_validate(response_to_validate, prompt=prompt):
                    return func_clean_up(response_to_validate, prompt=prompt)
                elif verbose: print(f"---- Legacy Safe repeat count: {i} ----\nValidation Failed Response: {response_to_validate}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            else:
                 print("Warning: No validation or cleanup function provided.")
                 return curr_gpt_response
        except Exception as e:
             error_info = f"Legacy Safe Generate Processing Error: {type(e).__name__} - {e}"
             metrics.fail_record(error_info)
             print(f"Attempt {i+1}/{repeat}: Error during legacy validation/cleanup - {error_info}")
             # import traceback; traceback.print_exc() # Uncomment for detailed stack trace

    print(f"Legacy safe_generate_response failed after {repeat} attempts.")
    return fail_safe_response


# ============================================================================
# ########################[SECTION 3: EMBEDDINGS] ###########################
# ============================================================================
local_embedding_model = None
local_embedding_model_name = DEFAULT_LOCAL_EMBEDDING_MODEL

def get_embedding(text, model="default"):
    global local_embedding_model
    """
    Generates embeddings using the appropriate backend based on key_type.
    Includes retry logic for Gemini quota errors & immediate exit for other Gemini API errors.
    """

    text = text.replace("\n", " ").strip()
    if not text:
        print("Warning: Attempting to embed empty or whitespace-only text.")
        return None

    # --- Check Embedding Pool Cache ---
    if use_embedding_pool:
        exist_embedding = get_embedding_pool(text)
        if exist_embedding is not None:
             if debug: print(f"Embedding cache hit for text: {text[:50]}...")
             return exist_embedding
        elif debug: print(f"Embedding cache miss for text: {text[:50]}...")

    # --- Handle 'llama' key_type using local Sentence Transformers ---
    if key_type == 'llama':
        if local_embedding_model is None:
             try:
                  # Load the model only once
                  from sentence_transformers import SentenceTransformer
                  print(f"--- Loading Local Embedding Model ({local_embedding_model_name}) ---")
                  # You might need to specify device='cuda' if you have a GPU and want to use it
                  local_embedding_model = SentenceTransformer(local_embedding_model_name)
             except ImportError:
                  print("ERROR: sentence-transformers library not found. Cannot generate local embeddings for 'llama' key_type.")
                  print("Install with: pip install sentence-transformers")
                  return None # Cannot proceed without the library
             except Exception as load_e:
                  print(f"ERROR: Failed to load local embedding model '{local_embedding_model_name}': {load_e}")
                  # Check if the model exists locally or needs downloading
                  return None # Cannot proceed without the model

        print(f"--- Generating Local Embedding ({local_embedding_model_name}) ---")
        try:
             start_time = time.time()
             # Generate embedding and convert numpy array to list
             embedding_vector = local_embedding_model.encode(text).tolist()
             time_use = time.time() - start_time
             print(f"Local embedding generated in {time_use:.2f}s")

             # Record metrics (optional, no token count for local models)
             # Use get_caller_function_names() if defined globally or imported
             metrics.call_record(get_caller_function_names(), local_embedding_model_name, 0, time_use)
             if use_embedding_pool: update_embedding_pool(text, embedding_vector)
             return embedding_vector
        except Exception as e:
             print(f"ERROR generating local embedding: {e}")
             metrics.fail_record(f"Local Embedding Error: {e}")
             return None # Return None on local embedding error
    # --- End 'llama' handling ---

    # --- Select Model Based on Backend if Default ---
    effective_model = model
    if effective_model == "default":
        # *** USE CENTRALIZED CONSTANTS ***
        if key_type == 'gemini':
            effective_model = DEFAULT_GEMINI_EMBEDDING_MODEL
        elif key_type == 'openai':
            effective_model = DEFAULT_OPENAI_EMBEDDING_MODEL
        elif key_type == 'azure':
            # Azure embedding often uses a specific deployment name
            effective_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", DEFAULT_AZURE_EMBEDDING_MODEL)
        elif key_type == 'llama':
             # Llama embedding support varies, using OpenAI default as placeholder
             effective_model = DEFAULT_OPENAI_EMBEDDING_MODEL
             print(f"Warning: Using default OpenAI embedding model '{effective_model}' for Llama.")
        else:
             raise ValueError(f"Unknown key_type '{key_type}' for default embedding model selection")
        # *** END USE CENTRALIZED CONSTANTS ***

    start_time = time.time()
    embedding_vector = None
    total_token = 0

    # --- GEMINI ---
    if key_type == 'gemini':
        if 'google.generativeai' not in sys.modules:
             raise RuntimeError("Gemini libraries not loaded.")

        print(f"--- Making Gemini Embedding Request ---")
        print(f"  Model: {effective_model}")

        for attempt in range(MAX_QUOTA_RETRIES):
            try:
                response = genai.embed_content(model=effective_model, content=text)
                embedding_vector = response['embedding']
                total_token = 0
                break # Exit loop on success

            except google.api_core.exceptions.ResourceExhausted as e:
                should_retry = _handle_quota_exception(e, attempt, MAX_QUOTA_RETRIES, effective_model)
                if not should_retry:
                     embedding_vector = None
                     total_token = 0
                     break

            except Exception as e: # Catch other API errors
                error_message = f"Embedding API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
                print(error_message)
                metrics.fail_record(error_message)
                # *** EXIT ON OTHER ERRORS ***
                print("Stopping the simulation due to unrecoverable API error.")
                sys.exit(1)
                # *** END EXIT ON OTHER ERRORS ***
                # embedding_vector = None # Unreachable
                # total_token = 0 # Unreachable
                # break # Unreachable
        # --- End Gemini Retry Loop ---

    # --- AZURE / OPENAI / LLAMA ---
    else:
        try:
            if key_type == 'azure':
                 print(f"--- Making Azure Embedding Request ---")
                 print(f"  Engine: {effective_model}") # Azure uses 'engine'
                 response = openai.Embedding.create(
                     api_base=openai_api_base, api_key=openai_api_key,
                     api_type=openai_api_type, api_version=openai_api_version,
                     input=[text], engine=effective_model # Use 'engine' for deployment name
                 )
                 embedding_vector = response['data'][0]['embedding']
                 total_token = response['usage']['total_tokens']

            elif key_type == 'openai':
                print(f"--- Making OpenAI Embedding Request ---")
                print(f"  Model: {effective_model}")
                response = openai.Embedding.create(
                    api_key=openai_api_key, api_base=openai_api_base,
                    input=[text], model=effective_model
                )
                embedding_vector = response['data'][0]['embedding']
                total_token = response['usage']['total_tokens']

            elif key_type == 'llama':
                 print(f"Warning: Embedding generation for 'llama' key_type is not standard via OpenAI API. Returning None.")
                 embedding_vector = None
                 total_token = 0

            else: # Should not happen
                raise ValueError(f"Unsupported key_type for embedding request: {key_type}")

        except Exception as e:
            error_message = f"Embedding API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
            print(error_message)
            metrics.fail_record(error_message)
            embedding_vector = None
            total_token = 0
            # Decide if non-Gemini errors should also exit immediately
            # print("Stopping the simulation due to unrecoverable API error.")
            # sys.exit(1)

    # --- Metrics and Caching ---
    function_name = get_caller_function_names()
    time_use = time.time() - start_time
    if embedding_vector is not None:
        metrics.call_record(function_name, effective_model, total_token, time_use)
        if use_embedding_pool:
             update_embedding_pool(text, embedding_vector)
    else:
         pass # Optionally record embedding failures

    return embedding_vector


# --- Main Execution Block (Example Usage) ---
if __name__ == '__main__':
    # (Keep example usage as is)
    print("\n--- Running Example Usage ---")
    print(f"Selected Key Type: {key_type}") # Show configured key type

    # --- Example 1: Basic Chat Request ---
    print("\nExample 1: Basic Chat Request (ChatGPT_request)")
    # This will use the default model for the selected key_type
    example_prompt_1 = "Explain the concept of a generative agent in simple terms."
    chat_response = ChatGPT_request(example_prompt_1)
    print(f"Prompt: {example_prompt_1}")
    print(f"Response:\n{chat_response}")

    # --- Example 2: Embedding Request ---
    print("\nExample 2: Embedding Request (get_embedding)")
    example_text_2 = "A generative agent simulates believable human behavior."
    embedding = get_embedding(example_text_2) # Uses default embedding model
    if embedding:
         print(f"Text: {example_text_2}")
         print(f"Embedding Dimension: {len(embedding)}")
         print(f"Embedding Snippet: {embedding[:5]}...")
    else:
         print(f"Failed to get embedding for: {example_text_2}")

    # --- Example 3: Safe JSON Generation ---
    print("\nExample 3: Safe JSON Generation (ChatGPT_safe_generate_response)")
    example_prompt_3 = "Describe a simple daily plan for a student named Alex. Output only the plan description."
    example_output_3 = "Wake up, attend class, study, have dinner, relax."
    special_instruction_3 = "Ensure the output is a single string containing the plan."

    def validate_plan(output, prompt):
        return isinstance(output, str) and len(output) > 10 # Simple validation

    def cleanup_plan(output, prompt):
        return output.strip()

    safe_response = ChatGPT_safe_generate_response(
        prompt=example_prompt_3,
        example_output=example_output_3,
        special_instruction=special_instruction_3,
        func_validate=validate_plan,
        func_clean_up=cleanup_plan,
        verbose=True # Enable verbose output for this example
    )
    print(f"Safe JSON Generation Result:\n{safe_response}")

