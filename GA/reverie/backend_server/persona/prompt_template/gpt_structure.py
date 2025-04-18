# /Users/mario/code/GitHub/gem-Generative-Agents/GA/reverie/backend_server/persona/prompt_template/gpt_structure.py

"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modifications for Gemini by: Your Name/AI Assistant
Modifications for Quota Retry by: Gemini

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

# Import configurations and utilities
from utils import * # Imports key_type, api keys/bases, etc.
from metrics import metrics
from pool import get_embedding_pool, update_embedding_pool

# --- Configure API Clients ---

# OpenAI/Azure/Llama (uses openai library)
# The key/base is set based on key_type within the request functions now,
# or relies on environment variables if openai_api_key is not explicitly passed.

# Google Gemini
if key_type == 'gemini':
    try:
        import google.generativeai as genai
        import google.api_core # For exception handling
        # Configure Gemini API key
        gemini_api_key_to_use = os.getenv("GOOGLE_API_KEY") or google_api_key # Prioritize env var
        if not gemini_api_key_to_use or gemini_api_key_to_use == "<Your Google API Key (from Google AI Studio or Cloud)>":
            raise ValueError("Gemini API Key not configured. Set GOOGLE_API_KEY environment variable or configure in utils.py")
        genai.configure(api_key=gemini_api_key_to_use)
        print("Gemini API configured.") # Confirmation log
    except ImportError:
        print("ERROR: 'google-generativeai' package not installed. Please install it to use the 'gemini' key_type.")
        # Optionally exit or raise a more specific error if Gemini is required
        sys.exit(1) # Exit if Gemini is selected but library not found
    except ValueError as e:
        print(e)
        # Optionally exit or raise
        sys.exit(1) # Exit if API key is not configured

# --- Helper Functions ---

def get_caller_function_names():
    """Gets the stack of caller function names."""
    stack = inspect.stack()
    # Start from index 2 to exclude get_caller_function_names and its caller
    caller_names = [frame.function for frame in stack][2:]
    return '.'.join(reversed(caller_names)) # Reverse for logical flow (outermost.inner...)

def temp_sleep(seconds=0.1):
    """Sleep for a specified duration."""
    if seconds <= 0:
        return
    time.sleep(seconds)

# --- Constants for Quota Retry ---
MAX_QUOTA_RETRIES = 3
BASE_RETRY_DELAY_SECONDS = 20 # Fallback delay if API doesn't suggest one

def _handle_quota_exception(e, attempt, max_retries, model_name):
    """Handles the ResourceExhausted exception with retry logic."""
    print(f"\n--- Gemini API Quota Limit Hit ({model_name} - Attempt {attempt + 1}/{max_retries}) ---")
    print(f"Error details: {e}")

    if attempt < max_retries - 1:
        # --- Calculate delay ---
        suggested_delay = BASE_RETRY_DELAY_SECONDS
        try:
            # Try to extract suggested delay from error metadata if available
            # google.api_core.exceptions.ResourceExhausted often has metadata
            if hasattr(e, 'metadata') and e.metadata:
                 for item in e.metadata:
                      if item.key == 'retry-delay': # Check common metadata keys
                           duration_str = item.value
                           # Example parsing '1m30s', adjust based on actual format
                           if 's' in duration_str:
                                suggested_delay = int(duration_str.split('s')[0]) # Simple seconds parsing
                           elif 'm' in duration_str:
                                suggested_delay = int(duration_str.split('m')[0]) * 60
                           suggested_delay = max(BASE_RETRY_DELAY_SECONDS, suggested_delay + 1) # Use suggested + buffer
                           break
            # Fallback: Check details attribute (less common for ResourceExhausted)
            elif hasattr(e, 'details') and e.details:
                 # Example: Parsing based on common string format (adjust if needed)
                 details_str = str(e.details)
                 if "retry_delay" in details_str:
                      match = re.search(r"seconds:\s*(\d+)", details_str)
                      if match:
                            suggested_delay = max(BASE_RETRY_DELAY_SECONDS, int(match.group(1)) + 1)
        except Exception as parse_err:
            print(f"Could not parse suggested retry delay from error metadata/details: {parse_err}")

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

# Unified Chat Model Request Function (Handles OpenAI, Azure, Llama, Gemini)
def llm_request(prompt,
                model_name="default", # Specify model ('gpt-3.5-turbo', 'gemini-pro', etc.)
                temperature=0.7, # Default temperature
                max_tokens=1024, # Default max tokens
                top_p=1.0,       # Default top_p
                stop_sequences=None, # Optional stop sequences
                time_sleep_second=0.1):
    """
    Handles requests to various LLM backends based on global `key_type`.
    Includes retry logic for Gemini quota errors.
    """
    temp_sleep(time_sleep_second)
    start_time = time.time()
    message = ""
    total_token = 0 # Default token count

    # --- Select Model Based on Backend if Default ---
    effective_model = model_name
    if effective_model == "default":
        if key_type == 'gemini':
            effective_model = "gemini-1.5-flash-latest" # Default Gemini model (using latest flash)
        elif key_type == 'openai' or key_type == 'azure' or key_type == 'llama':
            effective_model = "gpt-3.5-turbo" # Default for OpenAI compatible APIs
        else:
             raise ValueError(f"Unknown key_type '{key_type}' for default model selection")

    # --- GEMINI ---
    if key_type == 'gemini':
        if 'google.generativeai' not in sys.modules: # Check if import succeeded
             raise RuntimeError("Gemini libraries not loaded. Cannot make Gemini API call.")

        print(f"--- Making Gemini Request ---") # Debug log
        print(f"  Model: {effective_model}")
        print(f"  Prompt: {prompt[:100]}...") # Log snippet

        gemini_model = genai.GenerativeModel(effective_model)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
        safety_settings = [ # Adjust safety settings as needed
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        for attempt in range(MAX_QUOTA_RETRIES):
            try:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                if response.parts:
                    message = response.text
                else:
                    # Handle blocked responses (not quota errors)
                    message = f"ERROR: Gemini response blocked or empty. Reason: {response.prompt_feedback}"
                    print(message) # Log the block reason
                    # Decide if blocking should also cause exit or just return error message
                    # For now, return the error message
                total_token = 0 # Token count not directly available
                break # Exit loop on success or block

            except google.api_core.exceptions.ResourceExhausted as e:
                should_retry = _handle_quota_exception(e, attempt, MAX_QUOTA_RETRIES, effective_model)
                if not should_retry: # This path is taken if sys.exit was called
                     # Should not be reached, but as fallback:
                     message = f"ERROR: API Call Failed (Quota Exceeded after retries)"
                     total_token = 0
                     break # Exit loop

            except Exception as e:
                # Handle other API errors
                error_message = f"LLM API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
                print(error_message)
                metrics.fail_record(error_message) # Record the failure
                #message = f"ERROR: API Call Failed ({type(e).__name__})"
                print("Stopping the simulation due to unrecoverable API error.")
                sys.exit(1)

        # --- End Gemini Retry Loop ---

    # --- AZURE / LLAMA / OPENAI ---
    else:
        # (Keep existing logic for Azure, Llama, OpenAI - no quota retry needed here)
        try:
            # --- AZURE ---
            if key_type == 'azure':
                print(f"--- Making Azure Request ---")
                print(f"  Engine: {effective_model}")
                print(f"  Prompt: {prompt[:100]}...")
                completion = openai.ChatCompletion.create(
                    api_type=openai_api_type, api_version=openai_api_version,
                    api_base=openai_api_base, api_key=openai_api_key,
                    engine=effective_model, messages=[{"role": "user", "content": prompt}],
                    temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=stop_sequences
                )
                message = completion["choices"][0]["message"]["content"]
                total_token = completion['usage']['total_tokens']

            # --- LLAMA ---
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

            # --- OPENAI ---
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

            else:
                raise ValueError(f"Unsupported key_type for LLM request: {key_type}")

        except Exception as e:
            error_message = f"LLM API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
            print(error_message)
            metrics.fail_record(error_message)
            #message = f"ERROR: API Call Failed ({type(e).__name__})"
            #total_token = 0
            print("Stopping the simulation due to unrecoverable API error.")
            sys.exit(1)

    # --- Metrics Recording ---
    function_name = get_caller_function_names()
    time_use = time.time() - start_time
    # Record call only if it didn't end in an error message (or record separately?)
    if not message.startswith("ERROR:"):
        metrics.call_record(function_name, effective_model, total_token, time_use)

    return message # Return the final message (could be success or error string)


# ============================================================================
# #####################[SECTION 1: CHAT MODEL WRAPPERS] ######################
# ============================================================================

def ChatGPT_single_request(prompt, time_sleep_second=0.1):
    return llm_request(prompt, model_name="default", time_sleep_second=time_sleep_second)

def GPT4_request(prompt):
    """Requests GPT-4 or equivalent high-end model."""
    model_to_use = "gpt-4" # Default to GPT-4
    if key_type == 'gemini':
        model_to_use = "gemini-1.5-pro-latest" # Map to Gemini Pro
        print("Mapping GPT-4 request to Gemini model:", model_to_use)

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
    """ Requests the default chat model (GPT-3.5-turbo or Gemini-Flash). """
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
def LLM_safe_generate_response(request_func, # Pass the request function (GPT4_request or ChatGPT_request)
                               prompt,
                               example_output,
                               special_instruction,
                               repeat=3,
                               fail_safe_response="error",
                               func_validate=None,
                               func_clean_up=None,
                               verbose=False):
    """Generic safe generation function using a provided request function."""
    full_prompt = '"""\n' + prompt + '\n"""\n'
    full_prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    full_prompt += "Example output json:\n"
    full_prompt += json.dumps({"output": example_output}) # Use json.dumps for safety

    if verbose:
        print("----------- LLM PROMPT -----------")
        print(full_prompt)
        print("----------------------------------")

    for i in range(repeat):
        raw_response = request_func(full_prompt) # Call the passed request func

        if raw_response == "LLM ERROR": # Check for error returned by request func
             print(f"Attempt {i+1}/{repeat}: LLM request failed.")
             continue # Retry

        if verbose:
            print(f"--- Attempt {i+1}/{repeat} Raw Response ---")
            print(raw_response)
            print("-------------------------------")

        try:
            # Attempt to clean and parse the JSON part of the response
            start_index = raw_response.find('{')
            end_index = raw_response.rfind('}') + 1
            if start_index != -1 and end_index != -1:
                 json_str = raw_response[start_index:end_index]
                 parsed_json = json.loads(json_str)
                 # Check if 'output' key exists before accessing
                 if "output" in parsed_json:
                      curr_gpt_response = parsed_json["output"]
                 else:
                      print(f"Attempt {i+1}/{repeat}: 'output' key missing in parsed JSON.")
                      if verbose: print(f"Parsed JSON: {json_str}")
                      continue # Retry if key missing
            else:
                 print(f"Attempt {i+1}/{repeat}: Could not find valid JSON object in response.")
                 if verbose: print(f"Response was: {raw_response}")
                 continue # Retry

            # Validate and clean up the extracted output
            if func_validate and func_clean_up:
                if func_validate(curr_gpt_response, prompt=full_prompt):
                    return func_clean_up(curr_gpt_response, prompt=full_prompt)
                elif verbose:
                    print(f"Attempt {i+1}/{repeat}: Validation failed for output: {curr_gpt_response}")
            else:
                print("Warning: No validation or cleanup function provided to LLM_safe_generate_response.")
                return curr_gpt_response # Return parsed if no validation needed

        except json.JSONDecodeError as json_e:
            metrics.fail_record(f"JSON Parsing Failed: {json_e} in response: {raw_response[:500]}")
            print(f"Attempt {i+1}/{repeat}: JSON Parsing Error - {json_e}")
            if verbose: print(f"Raw response causing error: {raw_response}")
        except KeyError as key_e: # Should be caught by the 'in' check now, but keep for safety
            metrics.fail_record(f"KeyError '{key_e}' in parsed JSON: {json_str}")
            print(f"Attempt {i+1}/{repeat}: KeyError - '{key_e}' not found in JSON output.")
            if verbose: print(f"Parsed JSON causing error: {json_str}")
        except Exception as e:
            metrics.fail_record(f"Safe Generate Processing Error: {type(e).__name__} - {e}")
            print(f"Attempt {i+1}/{repeat}: Unexpected Error - {type(e).__name__}: {e}")

    print(f"LLM_safe_generate_response failed after {repeat} attempts.")
    return fail_safe_response


def GPT4_safe_generate_response(*args, **kwargs):
    """Safe generation using the GPT-4 request function."""
    return LLM_safe_generate_response(GPT4_request, *args, **kwargs)

def ChatGPT_safe_generate_response(*args, **kwargs):
    """Safe generation using the default ChatGPT request function."""
    return LLM_safe_generate_response(ChatGPT_request, *args, **kwargs)


# --- Legacy Safe Generate (Not Recommended for JSON) ---
def ChatGPT_safe_generate_response_OLD(prompt,
                                       repeat=3,
                                       fail_safe_response="error",
                                       func_validate=None,
                                       func_clean_up=None,
                                       verbose=False):
    """Older safe generation without explicit JSON structure assumption."""
    if verbose:
        print("----------- LEGACY CHATGPT PROMPT -----------")
        print(prompt)
        print("-------------------------------------------")

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
                 elif verbose:
                      print(f"---- Legacy repeat count: {i} ----")
                      print(f"Validation Failed Response: {response_to_validate}")
                      print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            else:
                 print("Warning: No validation or cleanup function provided to ChatGPT_safe_generate_response_OLD.")
                 return response_to_validate

        except Exception as e:
            # Log the specific error during validation/cleanup
            error_info = f"Legacy Safe Generate Processing Error: {type(e).__name__} - {e}"
            metrics.fail_record(error_info)
            print(f"Attempt {i+1}/{repeat}: Error during legacy validation/cleanup - {error_info}")
            # Log stack trace for debugging if needed
            # import traceback
            # traceback.print_exc()

    print("Legacy FAIL SAFE TRIGGERED")
    return fail_safe_response

# ============================================================================
# ###################[SECTION 2: LEGACY COMPLETION API] ###################
# ============================================================================

def gpt_request_all_version(prompt, gpt_parameter):
    """
    Handles requests for legacy OpenAI Completion models and attempts
    to map to Gemini's generate_content if key_type is 'gemini'.
    Includes retry logic for Gemini quota errors.
    """
    start_time = time.time()
    response_text = ""
    total_token = 0
    model_name = gpt_parameter.get("engine", gpt_parameter.get("model", "text-davinci-003"))

    # --- GEMINI (Attempted Mapping) ---
    if key_type == 'gemini':
        if 'google.generativeai' not in sys.modules:
             raise RuntimeError("Gemini libraries not loaded.")

        # Map legacy model name to a suitable Gemini model if needed
        # For simplicity, using flash as the default mapped model here
        mapped_gemini_model = "gemini-1.5-flash-latest"
        print(f"--- Making Gemini Request (Mapped from Legacy Completion) ---")
        print(f"  Legacy Model: {model_name} (mapped to {mapped_gemini_model})")
        print(f"  Prompt: {prompt[:100]}...")
        print(f"  Warning: Legacy frequency/presence penalty parameters ignored for Gemini.")

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
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )

                if response.parts:
                    response_text = response.text
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

            except Exception as e:
                error_message = f"Legacy Completion Map to Gemini ERROR ({mapped_gemini_model}): {type(e).__name__} - {e}"
                print(error_message)
                metrics.fail_record(error_message)
                response_text = "ERROR: API Call Failed"
                # total_token = 0
                # break 
                print("Stopping the simulation due to unrecoverable API error.")
                sys.exit(1)
        # --- End Gemini Retry Loop ---

    # --- AZURE / OPENAI (Legacy Completion Endpoint) ---
    else:
        # (Keep existing logic for Azure/OpenAI legacy completion)
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

    # --- Metrics Recording ---
    function_name = get_caller_function_names()
    time_use = time.time() - start_time
    if not response_text.startswith("ERROR:"):
        metrics.call_record(function_name, model_name, total_token, time_use)

    return response_text # Return only the text


def GPT_request(prompt, gpt_parameter):
    """Legacy wrapper for gpt_request_all_version."""
    temp_sleep() # Keep the original sleep call here
    try:
        response = gpt_request_all_version(prompt, gpt_parameter)
        # Check if the underlying function returned an error string
        if response == "ERROR: API Call Failed":
             # Distinguish between quota and other errors if possible, but keep original message
             print("TOKEN LIMIT EXCEEDED or other API Error")
             return "TOKEN LIMIT EXCEEDED" # Return original error message
        elif response == "ERROR: API Call Failed (Quota Exceeded after retries)":
             print("Quota limit exceeded after retries.")
             return "TOKEN LIMIT EXCEEDED" # Still return original error message for compatibility? Or a new one?
        return response
    except Exception as e:
        metrics.fail_record(f"GPT_request wrapper error: {e}")
        print(f"GPT_request ERROR (wrapper): {e}")
        print("TOKEN LIMIT EXCEEDED")
        return "TOKEN LIMIT EXCEEDED"

# --- Legacy Prompt Generation and Safe Execution ---
def generate_prompt(curr_input, prompt_lib_file):
    """Generates a prompt string from a template file and input(s)."""
    if isinstance(curr_input, str):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    try:
        # Ensure the path is correct relative to the execution context
        # Using absolute path or relative path from a known root might be safer
        # For now, assume prompt_lib_file path is correct
        with open(prompt_lib_file, "r") as f:
            prompt = f.read()
    except FileNotFoundError:
         print(f"Error: Prompt file not found at {prompt_lib_file}")
         # Depending on how critical this is, either raise error or return None/empty string
         # raise FileNotFoundError(f"Prompt file not found: {prompt_lib_file}")
         return "" # Return empty string if file not found

    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>", 1)[1]
    return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
    """Safe generation using the legacy GPT_request function."""
    if verbose:
        print("----------- LEGACY SAFE GEN PROMPT -----------")
        print(prompt)
        print("---------------------------------------------")

    # Use the general repeat count defined here, quota retries are handled inside GPT_request
    for i in range(repeat):
        curr_gpt_response = GPT_request(prompt, gpt_parameter)

        # Check for specific error messages returned by GPT_request
        if curr_gpt_response == "TOKEN LIMIT EXCEEDED":
            print(f"Attempt {i+1}/{repeat}: Legacy LLM request failed (TOKEN LIMIT EXCEEDED or Quota).")
            # If it was a quota error, GPT_request would have already exited.
            # If it was another error like token limit, continue retrying.
            continue
        elif curr_gpt_response == "ERROR: API Call Failed": # Generic error from underlying call
             print(f"Attempt {i+1}/{repeat}: Legacy LLM request failed (API Error).")
             continue


        try:
            if func_validate and func_clean_up:
                response_to_validate = curr_gpt_response # Already a string
                if func_validate(response_to_validate, prompt=prompt):
                    return func_clean_up(response_to_validate, prompt=prompt)
                elif verbose:
                    print(f"---- Legacy Safe repeat count: {i} ----")
                    print(f"Validation Failed Response: {response_to_validate}")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            else:
                 print("Warning: No validation or cleanup function provided to safe_generate_response.")
                 return curr_gpt_response

        except Exception as e:
             error_info = f"Legacy Safe Generate Processing Error: {type(e).__name__} - {e}"
             metrics.fail_record(error_info)
             print(f"Attempt {i+1}/{repeat}: Error during legacy validation/cleanup - {error_info}")
             # Optionally add traceback print here for debugging
             # import traceback
             # traceback.print_exc()

    print(f"Legacy safe_generate_response failed after {repeat} attempts.")
    return fail_safe_response


# ============================================================================
# ########################[SECTION 3: EMBEDDINGS] ###########################
# ============================================================================

def get_embedding(text, model="default"):
    """
    Generates embeddings using the appropriate backend based on key_type.
    Includes retry logic for Gemini quota errors.
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

    # --- Select Model Based on Backend if Default ---
    effective_model = model
    if effective_model == "default":
        if key_type == 'gemini':
            effective_model = "models/embedding-001" # Default Gemini embedding model
        elif key_type == 'openai' or key_type == 'azure' or key_type == 'llama':
            effective_model = "text-embedding-ada-002" # Default OpenAI/Azure
        else:
             raise ValueError(f"Unknown key_type '{key_type}' for default embedding model selection")

    start_time = time.time()
    embedding_vector = None
    total_token = 0 # Embedding token counts often handled differently

    # --- GEMINI ---
    if key_type == 'gemini':
        if 'google.generativeai' not in sys.modules:
             raise RuntimeError("Gemini libraries not loaded.")

        print(f"--- Making Gemini Embedding Request ---") # Debug log
        print(f"  Model: {effective_model}")

        for attempt in range(MAX_QUOTA_RETRIES):
            try:
                # Gemini's embed_content expects 'content'
                response = genai.embed_content(model=effective_model, content=text)
                embedding_vector = response['embedding']
                total_token = 0 # Token count not available
                break # Exit loop on success

            except google.api_core.exceptions.ResourceExhausted as e:
                should_retry = _handle_quota_exception(e, attempt, MAX_QUOTA_RETRIES, effective_model)
                if not should_retry:
                     embedding_vector = None # Ensure None if exited
                     total_token = 0
                     break

            except Exception as e:
                error_message = f"Embedding API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
                print(error_message)
                metrics.fail_record(error_message)
                embedding_vector = None # Return None on error
                # total_token = 0
                # break
                print("Stopping the simulation due to unrecoverable API error.")
                sys.exit(1)
        # --- End Gemini Retry Loop ---

    # --- AZURE / OPENAI / LLAMA ---
    else:
        # (Keep existing logic for Azure, OpenAI, Llama)
        try:
            if key_type == 'azure':
                 print(f"--- Making Azure Embedding Request ---")
                 print(f"  Engine: {effective_model}")
                 response = openai.Embedding.create(
                     api_base=openai_api_base, api_key=openai_api_key,
                     api_type=openai_api_type, api_version=openai_api_version,
                     input=[text], engine=effective_model
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

            else:
                raise ValueError(f"Unsupported key_type for embedding request: {key_type}")

        except Exception as e:
            error_message = f"Embedding API ERROR ({key_type}, Model: {effective_model}): {type(e).__name__} - {e}"
            print(error_message)
            metrics.fail_record(error_message)
            embedding_vector = None
            total_token = 0

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
    print(f"Selected Key Type: {key_type}")

    print("\nExample 1: Basic Chat Request (ChatGPT_request)")
    example_prompt_1 = "Explain the concept of a generative agent in simple terms."
    chat_response = ChatGPT_request(example_prompt_1)
    print(f"Prompt: {example_prompt_1}")
    print(f"Response:\n{chat_response}")

    print("\nExample 2: Embedding Request (get_embedding)")
    example_text_2 = "A generative agent simulates believable human behavior."
    embedding = get_embedding(example_text_2)
    if embedding:
         print(f"Text: {example_text_2}")
         print(f"Embedding Dimension: {len(embedding)}")
         print(f"Embedding Snippet: {embedding[:5]}...")
    else:
         print(f"Failed to get embedding for: {example_text_2}")

    print("\nExample 3: Safe JSON Generation (ChatGPT_safe_generate_response)")
    example_prompt_3 = "Describe a simple daily plan for a student named Alex. Output only the plan description."
    example_output_3 = "Wake up, attend class, study, have dinner, relax."
    special_instruction_3 = "Ensure the output is a single string containing the plan."
    def validate_plan(output, prompt): return isinstance(output, str) and len(output) > 10
    def cleanup_plan(output, prompt): return output.strip()
    safe_response = ChatGPT_safe_generate_response(
        prompt=example_prompt_3, example_output=example_output_3, special_instruction=special_instruction_3,
        func_validate=validate_plan, func_clean_up=cleanup_plan, verbose=True
    )
    print(f"Safe JSON Generation Result:\n{safe_response}")

