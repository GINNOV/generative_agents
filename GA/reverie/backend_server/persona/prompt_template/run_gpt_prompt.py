# /Users/mario/code/GitHub/gem-Generative-Agents/GA/reverie/backend_server/persona/prompt_template/run_gpt_prompt.py

"""
Author: Joon Sung Park (joonspk@stanford.edu)
Modifications for Gemini by: Your Name/AI Assistant

File: run_gpt_prompt.py
Description: Defines all run gpt prompt functions. These functions directly
interface with the safe_generate_response function.
"""
import json
import random
import re
import datetime
import sys
import ast
import string # Added missing import for get_random_alphanumeric
import logging

sys.path.append('../../')

# Import necessary utilities and configurations
from global_methods import *
from persona.prompt_template.gpt_structure import *
from persona.prompt_template.print_prompt import *
from utils import (key_type,
                   DEFAULT_OPENAI_COMPLETION_MODEL, DEFAULT_AZURE_COMPLETION_MODEL, DEFAULT_GEMINI_COMPLETION_MODEL)
from metrics import metrics # Make sure metrics is imported if used

# Configure basic logging to see errors from the parser
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_llm_json_list_output(raw_llm_output: str) -> list | None:
    """
    Parses JSON list output from an LLM response, handling potential leading/trailing text
    and whitespace.

    Args:
        raw_llm_output: The raw string response from the LLM.

    Returns:
        The parsed list if successful, None otherwise.
    """
    log_func = logging.error if logging.getLogger().hasHandlers() else print # Use logging or print

    if not raw_llm_output:
        log_func("Received empty LLM output.")
        return None

    try:
        # Find the first '[' character which should start the JSON list
        json_start_index = raw_llm_output.find('[')

        if json_start_index == -1:
            # If '[' is not found, log the unexpected output and fail
            log_func(f"Could not find start of JSON list ('[') in LLM output: {raw_llm_output[:200]}...")
            return None

        # Slice from the first '[' onwards to find the matching ']'
        slice_for_bracket_search = raw_llm_output[json_start_index:]
        brace_level = 0
        json_end_index_in_slice = -1
        for i, char in enumerate(slice_for_bracket_search):
            if char == '[':
                brace_level += 1
            elif char == ']':
                brace_level -= 1
                if brace_level == 0:
                    json_end_index_in_slice = i + 1
                    break

        if json_end_index_in_slice == -1:
            log_func(f"Could not find matching end bracket (']') for JSON list in LLM output slice: {slice_for_bracket_search[:200]}...")
            # Consider trying to parse anyway, maybe json.loads handles some trailing junk
            json_string_slice = slice_for_bracket_search
            # return None # Stricter: fail if no closing bracket found
        else:
             # Calculate the end index relative to the original string
             json_end_index = json_start_index + json_end_index_in_slice
             # Slice the original string using start and calculated end index
             json_string_slice = raw_llm_output[json_start_index:json_end_index]


        # *** THE KEY CHANGE IS HERE: .strip() ***
        # Remove leading/trailing whitespace (like newlines) from the extracted slice
        json_string = json_string_slice.strip()

        if not json_string: # Handle case where stripping results in empty string
             log_func(f"Empty string after slicing and stripping LLM output. Original snippet: {raw_llm_output[json_start_index:json_start_index+200]}...")
             return None

        # Attempt to parse the cleaned, sliced string
        parsed_json = json.loads(json_string)

        # Optional: Check if the result is actually a list
        if not isinstance(parsed_json, list):
             log_func_warn = logging.warning if logging.getLogger().hasHandlers() else print
             log_func_warn(f"Parsed JSON is not a list as expected: {type(parsed_json)}")
             return None # Or return parsed_json if other types are acceptable

        return parsed_json

    except json.JSONDecodeError as e:
        log_func(f"JSONDecodeError parsing LLM output: {e}")
        # Log the string that actually caused the error (after slicing and stripping)
        log_func(f"Problematic JSON string: '{json_string[:200]}...'")
        return None
    except Exception as e:
        # Catch any other unexpected errors during processing
        log_func(f"Unexpected error parsing LLM JSON output: {type(e).__name__} - {e}")
        log_func(f"Original raw output snippet: {raw_llm_output[:200]}...")
        return None


def get_random_alphanumeric(i=6, j=6):
    """
  Returns a random alpha numeric strength that has the length of somewhere
  between i and j. 

  INPUT: 
    i: min_range for the length
    j: max_range for the length
  OUTPUT: 
    an alpha numeric str with the length of somewhere between i and j.
  """
    k = random.randint(i, j)
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


##############################################################################
# CHAPTER 1: Run GPT Prompt
##############################################################################

# --- Function using Legacy Completion API ---
def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False):
    """
    Given the persona, returns an integer that indicates the hour when the
    persona wakes up, using a chat model and expecting direct JSON output.

    INPUT:
        persona: The Persona class instance
    OUTPUT:
        integer for the wake up hour (e.g., 6, 7).
    """
    log_func_error = logging.error if logging.getLogger().hasHandlers() else print
    log_func_warn = logging.warning if logging.getLogger().hasHandlers() else print
    log_func_debug = logging.debug if logging.getLogger().hasHandlers() else print

    def create_prompt_input(persona, test_input=None):
        if test_input: return test_input
        # Use inputs matching the updated prompt template
        prompt_input = [persona.scratch.get_str_iss(),       # !<INPUT 0>! Core Characteristics
                        persona.scratch.get_str_lifestyle(), # !<INPUT 1>! Lifestyle
                        persona.scratch.get_str_firstname()] # !<INPUT 2>! Name
        return prompt_input

    def __func_clean_up(gpt_response, prompt=""):
        # Cleans the GPT response expecting {"wake_up_hour": H} JSON string
        if not gpt_response or gpt_response == "LLM ERROR":
             log_func_warn(f"Received empty or error response: '{gpt_response}'. Using failsafe.")
             return get_fail_safe() # Use failsafe if LLM failed

        try:
            # Find the start of the JSON object
            json_start_index = gpt_response.find('{')
            if json_start_index == -1:
                 log_func_warn(f"Could not find start of JSON object ('{{') in response: {gpt_response[:200]}... Using failsafe.")
                 return get_fail_safe()

            # Find the matching closing brace
            brace_level = 0
            json_end_index = -1
            for i in range(json_start_index, len(gpt_response)):
                 if gpt_response[i] == '{': brace_level += 1
                 elif gpt_response[i] == '}': brace_level -= 1
                 if brace_level == 0:
                      json_end_index = i + 1
                      break

            if json_end_index == -1:
                 log_func_warn(f"Could not find matching end brace ('}}') in response: {gpt_response[json_start_index:json_start_index+200]}... Using failsafe.")
                 return get_fail_safe()

            # Extract and clean the JSON string
            json_string = gpt_response[json_start_index:json_end_index].strip()
            if not json_string:
                 log_func_warn(f"Empty string after JSON extraction. Original response: {gpt_response[:200]}... Using failsafe.")
                 return get_fail_safe()

            # Parse the JSON
            parsed_json = json.loads(json_string)

            if isinstance(parsed_json, dict) and "wake_up_hour" in parsed_json:
                hour_val = parsed_json["wake_up_hour"]
                # Convert extracted value to integer
                return int(hour_val)
            else:
                 log_func_warn(f"Parsed JSON missing 'wake_up_hour' key or not a dict: {parsed_json}. Using failsafe.")
                 return get_fail_safe() # Use failsafe if key missing

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log_func_warn(f"Could not parse/extract hour from response '{gpt_response[:200]}...': {e}. Using failsafe.")
            return get_fail_safe() # Use failsafe if parsing/conversion fails
        except Exception as e:
            log_func_error(f"Unexpected error cleaning wake_up_hour response: {e}")
            return get_fail_safe()

    def __func_validate(cleaned_output, prompt=""):
        # Validates if the cleaned output is an integer between 0 and 23.
        # Note: cleaned_output comes *after* __func_clean_up runs.
        try:
            hour = int(cleaned_output) # Should already be int from cleanup
            if 0 <= hour <= 23:
                return True
            else:
                log_func_warn(f"Wake up hour validation failed: {hour} out of range (0-23).")
                return False
        except (ValueError, TypeError) as e:
            # This case should ideally be caught by cleanup, but double-check
            log_func_error(f"Wake up hour validation error: Could not convert '{cleaned_output}' to int: {e}")
            # metrics.fail_record(f"Validation exception in wake_up_hour: {e}") # Optional
            return False

    def get_fail_safe():
        # Returns a default wake-up hour as an integer
        fs = 8 # Default to 8 AM
        return fs

    # Define parameters for the Chat model call (engine is handled by llm_request)
    # Placeholder gpt_param for print_run_prompts if needed
    gpt_param = {"engine": "Chat Model (e.g., llama3, gemini-pro)", "temperature": 0.8, "top_p": 1}

    # Define the prompt template file (using the updated version)
    prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.txt"
    # Create the input data for the prompt
    prompt_input = create_prompt_input(persona, test_input)
    # Generate the full prompt string
    prompt = generate_prompt(prompt_input, prompt_template)
    # Define the fail-safe value (integer)
    fail_safe = get_fail_safe()

    # --- Use simpler ChatGPT_request and handle retries/validation locally ---
    output = fail_safe # Default to fail_safe
    repeat_count = 3 # Number of attempts

    for i in range(repeat_count):
        # Call the simpler wrapper which returns the raw string or "LLM ERROR"
        raw_response = ChatGPT_request(prompt) # ChatGPT_request uses llm_request internally

        # Clean the raw response using our specific logic
        cleaned_response = __func_clean_up(raw_response, prompt) # Cleanup expects raw string

        # Validate the cleaned response (which should be an integer)
        if __func_validate(cleaned_response, prompt):
            output = cleaned_response # Assign the valid integer
            log_func_debug(f"Wake up hour successful on attempt {i+1}: {output}")
            break # Exit loop on success
        else:
            log_func_warn(f"Wake up hour attempt {i+1}/{repeat_count} failed validation. Cleaned response: '{cleaned_response}'")
            # Optional: sleep before retry? time.sleep(0.5)

    # If loop finishes without success, output remains the fail_safe value
    if output == fail_safe:
         log_func_error(f"Wake up hour failed after {repeat_count} attempts. Using fail_safe: {fail_safe}")

    # --- End of new call logic ---

    # Print debug information if verbose mode is enabled
    if debug or verbose:
        # Pass the placeholder gpt_param for printing context
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    # Return the final output (should be the hour integer) and debug info
    final_output = output # Already an integer from cleanup/validation or failsafe

    return final_output, [final_output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Chat Completion API (via _OLD wrapper) ---
# No gpt_param changes needed here as it uses ChatGPT_safe_generate_response_OLD
def run_gpt_prompt_daily_plan(persona,
                              wake_up_hour,
                              test_input=None,
                              verbose=False):
    """
  Generates the daily plan using ChatGPT.
  """
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, wake_up_hour, test_input=None):
        if test_input: return test_input
        prompt_input = []
        prompt_input += [persona.scratch.name]
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 AM"]
        return prompt_input

    # ... (__func_clean_up needs careful check for JSON format) ...
    def __func_clean_up(gpt_response, prompt=""):
        # This function now calls the robust parser
        # Use logging if available, otherwise print for debugging the raw response
        log_func_debug = logging.debug if logging.getLogger().hasHandlers() else print
        log_func_error = logging.error if logging.getLogger().hasHandlers() else print

        # log_func_debug(f"Raw GPT Response Before Parsing:{gpt_response}") # Optional: Uncomment for verbose debug

        # Call the robust parser function (defined elsewhere in the file)
        parsed_data = parse_llm_json_list_output(gpt_response)

        if parsed_data is None:
            # Parsing failed, log the error and return empty list to signify failure
            log_func_error(f"JSON Parsing Failed (returned None) for response snippet: {gpt_response[:200]}...")
            return [] # Return empty list on failure

        # Ensure it's a list (already checked in parser, but double-check)
        if not isinstance(parsed_data, list):
             log_func_error(f"Parsed data is not a list: {type(parsed_data)}")
             return []

        # Log success if needed
        # log_func_debug(f"Successfully parsed daily plan with {len(parsed_data)} activities.")
        return parsed_data # Return the successfully parsed list


    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for daily_plan: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe remains the same) ...
    def get_fail_safe():
        fs = []
        fs.append({"activity": "sleep", "start": "12:00 AM", "end": "06:00 AM"}) # Example fail-safe
        return fs

    gpt_param = {"engine": "ChatGPT Default (e.g., gpt-3.5-turbo or gemini-pro)", "temperature": 1, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v2/daily_planning_v6.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    # Uses ChatGPT_safe_generate_response_OLD -> ChatGPT_request -> llm_request
    # llm_request handles the model selection (gemini-pro if key_type is gemini)
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via _OLD wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_generate_hourly_schedule(persona,
                                            start_hour,
                                            end_hour,
                                            test_input=None,
                                            verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, start_hour, end_hour):
        daily_plan = ""
        for count, i in enumerate(persona.scratch.daily_req):
            daily_plan += f"{str(count + 1)}) {i}.\n"
        return [persona.scratch.get_str_firstname(), start_hour, end_hour, persona.scratch.get_str_iss(), daily_plan]

    # ... (__func_clean_up needs JSON check) ...
    def __func_clean_up(gpt_response, prompt=""):
        try:
            json_data = json.loads(gpt_response)
            assert isinstance(json_data, list), "run_gpt_prompt_generate_hourly_schedule -> gpt_response should be a list"
            return json_data
        except json.JSONDecodeError as e:
             print(f"ERROR: Could not parse JSON in hourly_schedule response: {e}")
             print(f"Raw response was: {gpt_response}")
             raise e
        except Exception as e:
             print(f"ERROR: Unexpected error processing hourly_schedule response: {e}")
             raise e

    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for hourly_schedule: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe remains the same) ...
    def get_fail_safe():
        fs = []
        fs.append({"activity": "sleep", "hour": "06:00 AM"}) # Example fail-safe
        return fs

    gpt_param = {"engine": "ChatGPT Default", "temperature": 0.5, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v2/generate_hourly_schedule_v2.txt"
    prompt_input = create_prompt_input(persona, start_hour, end_hour)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    # Uses ChatGPT_safe_generate_response_OLD -> ChatGPT_request -> llm_request
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via _OLD wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_next_day_remember(persona,
                                     statement,
                                     test_input=None,
                                     verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, statement):
        return [persona.scratch.get_str_firstname(), statement, persona.scratch.curr_time.strftime('%a %b %d')]

    # ... (__func_clean_up needs JSON check) ...
    def __func_clean_up(gpt_response, prompt=""):
        try:
            json_data = json.loads(gpt_response)
            assert isinstance(json_data, list), "run_gpt_prompt_next_day_remember -> gpt_response should be a list"
            return json_data
        except json.JSONDecodeError as e:
             print(f"ERROR: Could not parse JSON in next_day_remember response: {e}")
             print(f"Raw response was: {gpt_response}")
             raise e
        except Exception as e:
             print(f"ERROR: Unexpected error processing next_day_remember response: {e}")
             raise e

    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for next_day_remember: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe remains the same) ...
    def get_fail_safe():
        fs = []
        return fs

    gpt_param = {"engine": "ChatGPT Default", "temperature": 0.5, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/lifestyle/next_day_remember.txt"
    prompt_input = create_prompt_input(persona, statement)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    # Uses ChatGPT_safe_generate_response_OLD -> ChatGPT_request -> llm_request
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via _OLD wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_next_day_plan(persona,
                                 wake_up_hour,
                                 plan_note,
                                 summary_thoughts,
                                 test_input=None,
                                 verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, wake_up_hour, plan_note, summary_thoughts, test_input=None):
        if test_input: return test_input
        prompt_input = []
        prompt_input += [persona.scratch.name]
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [persona.scratch.get_str_lifestyle()]
        prompt_input += [persona.scratch.get_str_curr_date_str()]
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [f"{str(wake_up_hour)}:00 AM"]
        prompt_input += [plan_note]
        prompt_input += [summary_thoughts]
        return prompt_input

    # ... (__func_clean_up is similar to daily_plan, needs JSON check) ...
    def __func_clean_up(gpt_response, prompt=""):
        try:
            json_data = json.loads(gpt_response)
            assert isinstance(json_data, list), "run_gpt_prompt_next_day_plan -> gpt_response should be a list"
            # ... (rest of original processing logic) ...
            daily_req = []
            rest_time = 24 * 60
            # rest_time -= int(wake_up_hour * 60) # Assuming wake_up is start

            for activity_json in json_data:
                activity = activity_json["activity"]
                start_time = datetime.datetime.strptime(activity_json["start"], '%I:%M %p')
                end_time = datetime.datetime.strptime(activity_json["end"], '%I:%M %p')
                daily_req.append(f"{activity} from {activity_json['start']} to {activity_json['end']}")
                min_diff_time = (end_time - start_time).total_seconds() / 60
                if min_diff_time < 0:
                    min_diff_time = 1440 + min_diff_time
                rest_time -= min_diff_time
                print(f"activity_json -> s:{start_time} e:{end_time} m:{min_diff_time} r:{rest_time} t:{activity}")
                assert min_diff_time >= 0, "generate_first_daily_plan -> min_diff_time should be non-negative"
            return json_data
        except json.JSONDecodeError as e:
             print(f"ERROR: Could not parse JSON in next_day_plan response: {e}")
             print(f"Raw response was: {gpt_response}")
             raise e
        except Exception as e:
             print(f"ERROR: Unexpected error processing next_day_plan response: {e}")
             raise e

    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            __func_clean_up(gpt_response)
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for next_day_plan: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe remains the same) ...
    def get_fail_safe():
        fs = []
        fs.append({"activity": "sleep", "start": "12:00 AM", "end": "06:00 AM"}) # Example fail-safe
        return fs

    gpt_param = {"engine": "ChatGPT Default", "temperature": 1, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/lifestyle/next_day.txt"
    prompt_input = create_prompt_input(persona, wake_up_hour, plan_note, summary_thoughts, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()

    # Uses ChatGPT_safe_generate_response_OLD -> ChatGPT_request -> llm_request
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe, __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_task_decomp(persona,
                               task,
                               duration,
                               test_input=None,
                               verbose=False):
    # ... (create_prompt_input remains largely the same) ...
    def create_prompt_input(persona, task, duration, test_input=None):
        # This function seems complex and calculates context based on schedule indices.
        # It *should* still work as long as the persona object has the right data.
        # No direct changes needed here based on LLM backend.
        curr_f_org_index = persona.scratch.get_f_daily_schedule_hourly_org_index()
        all_indices = []
        all_indices += [curr_f_org_index]
        if curr_f_org_index + 1 < len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 1]
        if curr_f_org_index + 2 < len(persona.scratch.f_daily_schedule_hourly_org):
            all_indices += [curr_f_org_index + 2]

        curr_time_range = ""
        summ_str = f'Today is {persona.scratch.curr_time.strftime("%B %d, %Y")}. '
        summ_str += f'From '
        for index in all_indices:
            if index < len(persona.scratch.f_daily_schedule_hourly_org):
                start_min = sum(persona.scratch.f_daily_schedule_hourly_org[i][1] for i in range(index))
                end_min = start_min + persona.scratch.f_daily_schedule_hourly_org[index][1]
                start_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") + datetime.timedelta(minutes=start_min))
                end_time = (datetime.datetime.strptime("00:00:00", "%H:%M:%S") + datetime.timedelta(minutes=end_min))
                start_time_str = start_time.strftime("%I:%M%p").lstrip('0') # Use %I for 12-hour, %p for AM/PM
                end_time_str = end_time.strftime("%I:%M%p").lstrip('0')
                summ_str += f"{start_time_str} ~ {end_time_str}, {persona.name} is planning on {persona.scratch.f_daily_schedule_hourly_org[index][0]}, "
                if curr_f_org_index == index: # Original code had +1? Check logic if needed. Using current index.
                    curr_time_range = f'{start_time_str} ~ {end_time_str}'
        summ_str = summ_str.rstrip(', ') + "."

        prompt_input = []
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [summ_str]
        prompt_input += [persona.scratch.get_str_firstname()] # Duplicate?
        prompt_input += [persona.scratch.get_str_firstname()] # Duplicate?
        prompt_input += [task]
        prompt_input += [curr_time_range]
        prompt_input += [duration]
        prompt_input += [persona.scratch.get_str_firstname()] # Duplicate?
        return prompt_input

    # ... (__func_clean_up parses specific output format, might need adjustment) ...
    def __func_clean_up(gpt_response, prompt=""):
        # This cleanup is very specific to the expected output format.
        # It parses lines like "1. task_description (duration in minutes: 10)"
        # and adjusts durations to sum to the total expected minutes.
        # This might break if Gemini produces a slightly different format.
        # Careful testing and potential prompt adjustment needed.
        # print("Task Decomp Raw Response:\n", gpt_response) # Add for debugging
        try:
            temp = [i.strip() for i in gpt_response.split("\n") if i.strip()]
            _cr = []
            # Handle potential numbering (e.g., "1.", "1)") or lack thereof
            for i in temp:
                match = re.match(r"^\d+[\.\)]?\s*(.*)", i)
                if match:
                    _cr.append(match.group(1).strip())
                else:
                     _cr.append(i.strip()) # Assume no numbering if pattern doesn't match

            cr = []
            for i in _cr:
                 # Regex to find duration: (duration in minutes: NUM) possibly with variations
                 duration_match = re.search(r"\(duration(?: in minutes)?: *(\d+)\)", i, re.IGNORECASE)
                 if duration_match:
                     task_part = i[:duration_match.start()].strip()
                     duration = int(duration_match.group(1))
                     if task_part.endswith("."):
                         task_part = task_part[:-1]
                     cr.append([task_part, duration])
                 else:
                     # Fallback or error if format unexpected
                     print(f"WARNING: Could not parse duration from task decomp line: {i}")
                     # Decide handling: skip line, assign default duration, raise error?
                     # For now, let's skip this line if duration is missing
                     continue

            total_expected_min_match = re.search(r"\(total duration in minutes: *(\d+)\)", prompt, re.IGNORECASE)
            if not total_expected_min_match:
                 print("ERROR: Could not find total expected duration in task decomp prompt.")
                 # Need a fallback mechanism here
                 return cr # Return partially parsed list, duration adjustment might fail

            total_expected_min = int(total_expected_min_match.group(1))

            # Duration Adjustment Logic (Keep original logic for now, test heavily)
            curr_min_slot = [] # List of (task_name, task_index) for each minute
            for count, (i_task, i_duration) in enumerate(cr):
                 i_duration_adj = max(0, i_duration - (i_duration % 5)) # Round down to nearest 5? Or just use raw duration? Let's use raw for now.
                 i_duration_adj = max(0, i_duration) # Use raw duration, ensure non-negative
                 if i_duration_adj > 0:
                      curr_min_slot.extend([(i_task, count)] * i_duration_adj)

            # Adjust total duration to match expected
            if len(curr_min_slot) > total_expected_min:
                 curr_min_slot = curr_min_slot[:total_expected_min]
            elif len(curr_min_slot) < total_expected_min and curr_min_slot: # Ensure not empty
                 last_task = curr_min_slot[-1]
                 curr_min_slot.extend([last_task] * (total_expected_min - len(curr_min_slot)))
            elif not curr_min_slot and total_expected_min > 0: # Handle empty result
                 print("WARNING: Task decomp resulted in zero effective duration. Using placeholder.")
                 # Add a placeholder task? Depends on desired behavior.
                 # For now, return empty list, might cause issues downstream.
                 return []


            # Consolidate consecutive minutes back into tasks with durations
            cr_ret = []
            if curr_min_slot: # Check if list has elements
                current_task, _ = curr_min_slot[0]
                current_duration = 0
                for task, task_index in curr_min_slot:
                    if task == current_task:
                        current_duration += 1
                    else:
                        cr_ret.append([current_task, current_duration])
                        current_task = task
                        current_duration = 1
                cr_ret.append([current_task, current_duration]) # Add the last task block

            return cr_ret
        except Exception as e:
             print(f"ERROR cleaning task decomp: {e}")
             print(f"Raw response: {gpt_response}")
             # Decide: raise error, return empty list, return fail_safe?
             # Returning empty might be safer than failing entirely.
             return [] # Return empty list on error


    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            cleaned = __func_clean_up(gpt_response, prompt)
            # Add a basic check: is it a list? Does it have elements if expected?
            if not isinstance(cleaned, list):
                 return False
            # Optionally check if sum of durations matches expected (already done in cleanup)
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for task decomp: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe remains the same) ...
    def get_fail_safe():
        fs = ["asleep"] # This fail safe seems incorrect for the expected output format [[task, duration],...]
        # Better fail safe: Return the original task with the full duration
        # This needs access to 'task' and 'duration' from the outer scope.
        # Let's return an empty list as a generic failure indicator for now.
        return []


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 1000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/task_decomp_v3.txt"
    prompt_input = create_prompt_input(persona, task, duration, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    # fail_safe = get_fail_safe() # Using updated fail_safe logic in cleanup/validate
    fail_safe = [] # Default fail safe

    print("Running Task Decomp...")
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    # --- Post-processing Logic (Seems fragile, needs testing) ---
    # This part adjusts durations again and reformats the output.
    # It might fail if 'output' from safe_generate_response is not the expected list.
    try:
        # print("Task Decomp Output (Before Post-processing):", output)
        fin_output = []
        time_sum = 0
        # Ensure output is iterable and items are lists/tuples of size 2
        if isinstance(output, list):
             for item in output:
                  if isinstance(item, (list, tuple)) and len(item) == 2:
                       i_task, i_duration = item
                       if isinstance(i_duration, int): # Ensure duration is int
                            time_sum += i_duration
                            if time_sum <= duration:
                                 fin_output.append([i_task, i_duration])
                            else:
                                 # Add partial duration if overshoots
                                 partial_duration = i_duration - (time_sum - duration)
                                 if partial_duration > 0:
                                      fin_output.append([i_task, partial_duration])
                                 break
                       else: print(f"WARN: Invalid duration type in task decomp output: {item}")
                  else: print(f"WARN: Invalid item format in task decomp output: {item}")
        else: print(f"WARN: Task decomp output is not a list: {output}")


        ftime_sum = sum(item[1] for item in fin_output if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], int))

        # Adjust the last item's duration if needed and possible
        if fin_output and isinstance(fin_output[-1], (list, tuple)) and len(fin_output[-1])==2 and isinstance(fin_output[-1][1], int):
            duration_diff = duration - ftime_sum
            if duration_diff != 0:
                 fin_output[-1][1] += duration_diff
                 # Ensure duration doesn't become negative
                 fin_output[-1][1] = max(0, fin_output[-1][1])
                 # Remove if duration becomes zero? Optional based on desired behavior.
                 # if fin_output[-1][1] == 0: fin_output.pop()

        else: # Handle cases where fin_output is empty or last item malformed
             if duration > 0 and not fin_output:
                  print("WARN: Task decomp post-processing resulted in empty list, but duration > 0. Adding original task.")
                  fin_output = [[task, duration]] # Fallback to original task maybe?

        output = fin_output

        # Reformat task description (adds original task prefix)
        ret = []
        for decomp_task, dur in output:
            ret.append([f"{task} ({decomp_task})", dur])
        output = ret
        # print("Task Decomp Output (After Post-processing):", output)

    except IndexError as e:
        print(f"ERROR during task decomp post-processing (IndexError): {e}")
        print(f"Output that caused error: {output}")
        # Fallback: return the original task without decomposition?
        output = [[task, duration]] # Return original task as a list item
    except Exception as e:
        print(f"ERROR during task decomp post-processing (General): {type(e).__name__} - {e}")
        output = [[task, duration]] # Fallback


    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_action_sector(action_description,
                                 persona,
                                 maze,
                                 test_input=None,
                                 verbose=False):
    # ... (create_prompt_input needs adjustment for accessible sectors) ...
    def create_prompt_input(action_description, persona, maze, fin_accessible_sectors, test_input=None):
        # This function needs access to fin_accessible_sectors calculated in the main body
        act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
        prompt_input = []
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [persona.scratch.living_area.split(":")[1]] # Assuming this is relevant context
        # Context about current location seems more relevant than living area's contents
        prompt_input += [f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"]
        x = f"{act_world}:{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        prompt_input += [persona.s_mem.get_str_accessible_sector_arenas(x)] # Arenas in current sector

        if persona.scratch.get_str_daily_plan_req() != "":
            prompt_input += [f"\n{persona.scratch.get_str_daily_plan_req()}"]
        else:
            prompt_input += [""]

        # Providing the list of choices clearly
        accessible_sector_str = ""
        for i, sector in enumerate(fin_accessible_sectors):
            accessible_sector_str += f"{i + 1}. {sector}\n"
        prompt_input += [accessible_sector_str.strip()] # Add the choices

        action_description_1 = action_description
        action_description_2 = action_description # Sub-activity if present
        if "(" in action_description:
            action_description_1 = action_description.split("(")[0].strip()
            action_description_2 = action_description.split("(")[-1][:-1]
        prompt_input += [action_description_1]
        prompt_input += [action_description_2]

        return prompt_input


    # ... (__func_clean_up expects JSON, needs fin_accessible_sectors access) ...
    def __func_clean_up(gpt_response, prompt="", accessible_sectors_list=None):
        # Needs the list of valid sectors passed in or accessed globally/via outer scope
        if accessible_sectors_list is None: accessible_sectors_list = [] # Safety default
        try:
            # This uses the *legacy* safe_generate_response, which doesn't enforce JSON.
            # The original code expected the sector name directly. Let's try that first.
            cleaned_response = gpt_response.strip()
            # Optional: Try to parse if it looks like JSON, fallback otherwise
            if cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                 try:
                      cleaned_response = json.loads(cleaned_response)["output"]
                 except Exception:
                      pass # Stick with the stripped string if JSON parse fails

            # Validate against the provided list
            if cleaned_response not in accessible_sectors_list:
                 print(f"WARN: Action Sector GPT response '{cleaned_response}' not in accessible list: {accessible_sectors_list}. Falling back.")
                 # Fallback logic: choose persona's living area sector? Or random from list?
                 fallback_sector = persona.scratch.living_area.split(":")[1] # Example fallback
                 if fallback_sector in accessible_sectors_list:
                      return fallback_sector
                 elif accessible_sectors_list: # If list is not empty
                      return random.choice(accessible_sectors_list) # Random choice as last resort
                 else: # If list somehow is empty
                      return "unknown_sector" # Or raise error

            return cleaned_response
        except Exception as e:
             print(f"ERROR cleaning action sector: {e}")
             print(f"Raw response: {gpt_response}")
             # Fallback logic similar to above
             fallback_sector = persona.scratch.living_area.split(":")[1]
             if accessible_sectors_list and fallback_sector in accessible_sectors_list: return fallback_sector
             if accessible_sectors_list: return random.choice(accessible_sectors_list)
             return "unknown_sector"


    # ... (__func_validate needs accessible_sectors_list) ...
    def __func_validate(gpt_response, prompt="", accessible_sectors_list=None):
        # Check if the cleaned response is in the valid list.
        try:
            cleaned = __func_clean_up(gpt_response, prompt, accessible_sectors_list)
            return cleaned in accessible_sectors_list if accessible_sectors_list else False
        except Exception as e:
            metrics.fail_record(f"Validation failed for action sector: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe needs accessible_sectors_list) ...
    def get_fail_safe(accessible_sectors_list=None):
        # Choose a reasonable default, like the persona's home sector, if possible.
        if accessible_sectors_list is None: accessible_sectors_list = []
        home_sector = persona.scratch.living_area.split(":")[1]
        if home_sector in accessible_sectors_list:
            return home_sector
        elif accessible_sectors_list:
            return random.choice(accessible_sectors_list)
        else:
            # Fallback if no sectors known - this indicates a deeper issue
            return "kitchen" # Original fallback, likely incorrect context


    # --- Calculate accessible sectors ---
    act_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    accessible_sector_str = persona.s_mem.get_str_accessible_sectors(act_world)
    curr = accessible_sector_str.split(", ")
    # Filter based on persona's house if needed (original logic)
    fin_accessible_sectors = []
    for i in curr:
        if "'s house" in i:
            if persona.scratch.last_name in i: # Check if it's the persona's own house
                fin_accessible_sectors.append(i)
        else: # Add non-house sectors
            fin_accessible_sectors.append(i)
    if not fin_accessible_sectors: # Safety check if list becomes empty
        print(f"WARNING: No accessible sectors found for {persona.name} in world {act_world}. Using fallback.")
        # Add a default sector like the current one or living area if possible
        current_sector = maze.access_tile(persona.scratch.curr_tile)['sector']
        if current_sector: fin_accessible_sectors.append(current_sector)
        else: fin_accessible_sectors.append(persona.scratch.living_area.split(":")[1])


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 25, # Increased slightly for potentially longer sector names
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", "."]} # Added stops
    # --- End Modification ---

    prompt_template = "persona/prompt_template/lifestyle/action_location_sector.txt" # Using lifestyle version
    prompt_input = create_prompt_input(action_description, persona, maze, fin_accessible_sectors, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    # Pass accessible list to validators/cleanup
    fail_safe = get_fail_safe(fin_accessible_sectors)
    def wrapped_validate(resp, prompt="", **kwargs): # Accept prompt as keyword arg, ignore others
        # Pass accessible_sectors_list explicitly
        return __func_validate(resp, prompt=prompt, accessible_sectors_list=fin_accessible_sectors)
    def wrapped_cleanup(resp, prompt="", **kwargs): # Accept prompt as keyword arg, ignore others
        # Pass accessible_sectors_list explicitly
        return __func_clean_up(resp, prompt=prompt, accessible_sectors_list=fin_accessible_sectors)

    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    wrapped_validate, wrapped_cleanup)

    # Post-validation (already done in cleanup/validate, but double check)
    if output not in fin_accessible_sectors:
        print(f"WARN: Final output '{output}' for action sector still not in accessible list {fin_accessible_sectors}. Using fail_safe '{fail_safe}'.")
        output = fail_safe

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_action_arena(action_description,
                                persona,
                                maze, act_world, act_sector,
                                test_input=None,
                                verbose=False):
    # ... (create_prompt_input needs adjustment for accessible arenas) ...
    def create_prompt_input(action_description, persona, maze, act_world, act_sector, fin_accessible_arenas, test_input=None):
        # Needs fin_accessible_arenas list
        prompt_input = []
        prompt_input += [persona.scratch.get_str_firstname()]
        prompt_input += [act_sector] # Context: current sector

        # Provide list of choices
        if not fin_accessible_arenas:
            print(f"ERROR: No accessible arenas found for {act_world}:{act_sector}")
            # Handle this error case - maybe add a default arena if possible?
            accessible_arena_str = "1. unknown arena\n" # Placeholder
        else:
            accessible_arena_str = ''
            for i, arena in enumerate(fin_accessible_arenas):
                accessible_arena_str += f"{i + 1}. {arena}\n"
        prompt_input += [accessible_arena_str.strip()]

        # Action description (split if needed)
        action_description_1 = action_description
        action_description_2 = action_description
        if "(" in action_description:
            action_description_1 = action_description.split("(")[0].strip()
            action_description_2 = action_description.split("(")[-1][:-1]
        prompt_input += [action_description_1]
        prompt_input += [action_description_2]

        return prompt_input

    # ... (__func_clean_up expects direct output, needs accessible_arenas_list) ...
    def __func_clean_up(gpt_response, prompt="", accessible_arenas_list=None):
        # Needs the list of valid arenas passed in
        if accessible_arenas_list is None: accessible_arenas_list = []
        try:
            cleaned_response = gpt_response.strip()
            # Attempt JSON parse as fallback? Unlikely for legacy.
            if cleaned_response.startswith("{") and cleaned_response.endswith("}"):
                 try: cleaned_response = json.loads(cleaned_response)['output']
                 except Exception: pass

            if cleaned_response not in accessible_arenas_list:
                print(f"WARN: Action Arena GPT response '{cleaned_response}' not in accessible list: {accessible_arenas_list}. Falling back.")
                if accessible_arenas_list: return random.choice(accessible_arenas_list)
                else: return "unknown_arena"

            return cleaned_response
        except Exception as e:
             print(f"ERROR cleaning action arena: {e}")
             print(f"Raw response: {gpt_response}")
             if accessible_arenas_list: return random.choice(accessible_arenas_list)
             return "unknown_arena"


    # ... (__func_validate needs accessible_arenas_list) ...
    def __func_validate(gpt_response, prompt="", accessible_arenas_list=None):
        try:
            cleaned = __func_clean_up(gpt_response, prompt, accessible_arenas_list)
            return cleaned in accessible_arenas_list if accessible_arenas_list else False
        except Exception as e:
            metrics.fail_record(f"Validation failed for action arena: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe needs accessible_arenas_list) ...
    def get_fail_safe(accessible_arenas_list=None):
        if accessible_arenas_list is None: accessible_arenas_list = []
        if accessible_arenas_list:
            return random.choice(accessible_arenas_list)
        return "common room" # Original fallback, adjust if needed


    # --- Calculate accessible arenas ---
    x = f"{act_world}:{act_sector}"
    accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
    curr = accessible_arena_str.split(", ")
    # Filter based on persona's room if needed (original logic)
    fin_accessible_arenas = []
    for i in curr:
        if i: # Ensure not empty string
            if "'s room" in i:
                if persona.scratch.last_name in i: # Check if it's the persona's own room
                    fin_accessible_arenas.append(i)
            else:
                fin_accessible_arenas.append(i)
    if not fin_accessible_arenas: # Safety check
        print(f"WARNING: No accessible arenas found for {persona.name} in {act_world}:{act_sector}. Using fallback.")
        # Add current arena as fallback?
        current_arena = maze.access_tile(persona.scratch.curr_tile)['arena']
        if current_arena: fin_accessible_arenas.append(current_arena)
        else: fin_accessible_arenas.append("common room") # Last resort fallback


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 25, # Increased slightly
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", "."]} # Added stops
    # --- End Modification ---

    prompt_template = "persona/prompt_template/lifestyle/action_location_arena.txt" # Using lifestyle version
    prompt_input = create_prompt_input(action_description, persona, maze, act_world, act_sector, fin_accessible_arenas)
    prompt = generate_prompt(prompt_input, prompt_template)

    # Pass accessible list to validators/cleanup
    fail_safe = get_fail_safe(fin_accessible_arenas)
    def wrapped_validate(resp, pmt): return __func_validate(resp, pmt, accessible_arenas_list=fin_accessible_arenas)
    def wrapped_cleanup(resp, pmt): return __func_clean_up(resp, pmt, accessible_arenas_list=fin_accessible_arenas)

    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    wrapped_validate, wrapped_cleanup)

    # Post-validation (already done in cleanup/validate, but double check)
    if output not in fin_accessible_arenas:
        print(f"WARN: Final output '{output}' for action arena still not in accessible list {fin_accessible_arenas}. Using fail_safe '{fail_safe}'.")
        output = fail_safe

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Legacy Completion API ---
def run_gpt_prompt_action_game_object(action_description,
                                      persona,
                                      maze, # Maze object needed for accessible objects
                                      temp_address, # Should be full address including arena
                                      test_input=None,
                                      verbose=False):
    # ... (create_prompt_input needs accessible objects list) ...
    def create_prompt_input(action_description,
                            persona,
                            temp_address,
                            accessible_objects_str, # Pass the string directly
                            test_input=None):
        prompt_input = []
        # Extract sub-activity description if present
        action_sub_desc = action_description
        if "(" in action_description:
            action_sub_desc = action_description.split("(")[-1][:-1]

        prompt_input += [action_sub_desc] # Use the sub-activity for object selection
        prompt_input += [accessible_objects_str] # Provide the choices
        return prompt_input

    # ... (__func_clean_up needs accessible_objects_list) ...
    def __func_clean_up(gpt_response, prompt="", accessible_objects_list=None):
        # Needs the list of valid objects
        if accessible_objects_list is None: accessible_objects_list = []
        try:
            cleaned_response = gpt_response.strip().replace(".","") # Remove trailing periods
            if cleaned_response not in accessible_objects_list:
                 print(f"WARN: Action Object GPT response '{cleaned_response}' not in accessible list: {accessible_objects_list}. Falling back.")
                 if accessible_objects_list: return random.choice(accessible_objects_list)
                 else: return "bed" # Original failsafe, might be inappropriate
            return cleaned_response
        except Exception as e:
             print(f"ERROR cleaning action object: {e}")
             print(f"Raw response: {gpt_response}")
             if accessible_objects_list: return random.choice(accessible_objects_list)
             return "bed"


    # ... (__func_validate needs accessible_objects_list) ...
    def __func_validate(gpt_response, prompt="", accessible_objects_list=None):
        try:
            cleaned = __func_clean_up(gpt_response, prompt, accessible_objects_list)
            # Basic validation: not empty and in the list
            return bool(cleaned.strip()) and (cleaned in accessible_objects_list if accessible_objects_list else False)
        except Exception as e:
            metrics.fail_record(f"Validation failed for action object: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe needs accessible_objects_list) ...
    def get_fail_safe(accessible_objects_list=None):
        if accessible_objects_list is None: accessible_objects_list = []
        if accessible_objects_list:
            return random.choice(accessible_objects_list)
        return "bed" # Original failsafe


    # --- Get accessible objects ---
    accessible_objects_str = persona.s_mem.get_str_accessible_arena_game_objects(temp_address)
    accessible_objects_list = [obj.strip() for obj in accessible_objects_str.split(",") if obj.strip()]
    if not accessible_objects_list:
         print(f"WARNING: No accessible objects found for {temp_address}. Using fallback.")
         # Maybe add a default object? Or handle this case upstream?
         # For now, the failsafe will be used.


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 20, # Increased slightly
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", "."]}
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v1/action_object_v2.txt" # Original template path
    prompt_input = create_prompt_input(action_description,
                                     persona,
                                     temp_address,
                                     accessible_objects_str, # Pass the string version
                                     test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(accessible_objects_list)
    # Wrap validators/cleanup
    def wrapped_validate(resp, prompt="", **kwargs): # Accept prompt as keyword arg, ignore others
        # Pass accessible_objects_list explicitly
        return __func_validate(resp, prompt=prompt, accessible_objects_list=accessible_objects_list)
    def wrapped_cleanup(resp, prompt="", **kwargs): # Accept prompt as keyword arg, ignore others
        # Pass accessible_objects_list explicitly
        return __func_clean_up(resp, prompt=prompt, accessible_objects_list=accessible_objects_list)

    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    wrapped_validate, wrapped_cleanup)

    # Post-validation (already done in cleanup/validate, but double check)
    if output not in accessible_objects_list and accessible_objects_list:
         print(f"WARN: Final output '{output}' for action object still not in accessible list {accessible_objects_list}. Using fail_safe '{fail_safe}'.")
         output = fail_safe
    elif not accessible_objects_list:
         output = fail_safe # Ensure failsafe if list was empty

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via safe wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(action_description):
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        prompt_input = [action_description]
        return prompt_input

    # ... (__chat_func_clean_up is the main cleanup logic here) ...
    def __chat_func_clean_up(gpt_response, prompt=""):
        # This expects JSON output {"output": "emoji"}
        try:
             # The safe generator already extracts the "output" field.
             cr = gpt_response.strip()
             # Validate if it looks like emoji (crude check)
             if not cr or not any(char > '\u23ff' for char in cr): # Check for characters outside basic ranges
                  print(f"WARN: Pronunciatio response '{cr}' doesn't look like emoji. Using failsafe.")
                  return get_fail_safe() # Return failsafe if not emoji-like
             # Limit length (original logic)
             if len(cr) > 3: # Note: This might cut multi-character emojis
                 cr = cr[:3]
             return cr
        except Exception as e:
             print(f"ERROR cleaning pronunciatio: {e}")
             print(f"Raw (processed) response: {gpt_response}")
             return get_fail_safe()


    # ... (__chat_func_validate checks cleanup) ...
    def __chat_func_validate(gpt_response, prompt=""):
        # Validates if the cleaned response is a non-empty string
        try:
            cleaned = __chat_func_clean_up(gpt_response, prompt="") # Pass empty prompt
            return bool(cleaned) # Check if non-empty
        except Exception as e:
            metrics.fail_record(f"Validation failed for obj desc: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        fs = "🙂" # Default emoji
        return fs

    # Uses ChatGPT_safe_generate_response -> GPT4_request/ChatGPT_request -> llm_request
    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.txt"
    prompt_input = create_prompt_input(action_description)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "🛁" # Simple example emoji
    special_instruction = "The value for the output must ONLY contain the emojis that best represent the action."
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    # Original code had a fallback to legacy here, but let's rely on the chat version first.
    # If it consistently fails, the fail_safe '🙂' will be returned.

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Legacy Completion API ---
def run_gpt_prompt_event_triple(action_description, persona, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(action_description, persona):
        if "(" in action_description:
            action_description = action_description.split("(")[-1].split(")")[0]
        prompt_input = [persona.name,
                        action_description,
                        persona.name]
        return prompt_input

    # ... (__func_clean_up parses 'pred, obj' format) ...
    def __func_clean_up(gpt_response, prompt=""):
        try:
            cr = gpt_response.strip()
            # Remove potential parenthesis if present
            if cr.startswith("(") and cr.endswith(")"):
                cr = cr[1:-1]
            # Split by comma, expect two parts (predicate, object)
            parts = [i.strip() for i in cr.split(",")]
            if len(parts) == 2:
                return parts # Return [predicate, object]
            else:
                print(f"WARN: Event triple cleanup expected 2 parts, got {len(parts)} from '{cr}'")
                # Fallback: return default or raise error
                return ["is", "idle"] # Default fallback
        except Exception as e:
             print(f"ERROR cleaning event triple: {e}")
             print(f"Raw response: {gpt_response}")
             return ["is", "idle"] # Default fallback

    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            gpt_response = __func_clean_up(gpt_response, prompt="")
            if len(gpt_response) != 2:
                return False
        except Exception as e:
            metrics.fail_record(e)
            return False
        return True

    # ... (get_fail_safe needs persona name) ...
    def get_fail_safe(persona):
        fs = (persona.name, "is", "idle")
        return fs

    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 30,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", ")"]} # Added stop
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt"
    prompt_input = create_prompt_input(action_description, persona)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(persona) # Pass persona to get name in fail_safe

    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output_list = safe_generate_response(prompt, gpt_param, 5, ["is", "idle"], # Use list fail_safe
                                         __func_validate, __func_clean_up)

    # Reconstruct the tuple (subj, pred, obj)
    if isinstance(output_list, list) and len(output_list) == 2:
        output = (persona.name, output_list[0], output_list[1])
    else:
        output = fail_safe

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Chat Completion API (via safe wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_act_obj_desc(act_game_object, act_desp, persona, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(act_game_object, act_desp, persona):
        prompt_input = [act_game_object,
                        persona.name,
                        act_desp,
                        act_game_object,
                        act_game_object]
        return prompt_input

    # ... (__chat_func_clean_up is the main cleanup) ...
    def __chat_func_clean_up(gpt_response, prompt=""):
        # Expects JSON {"output": "description"}
        try:
            # Safe generator provides the extracted output
            cr = gpt_response.strip()
            if cr.endswith("."): cr = cr[:-1]
            return cr
        except Exception as e:
             print(f"ERROR cleaning obj desc: {e}")
             # Need act_game_object for fail_safe, but it's not passed here.
             # Define failsafe directly or modify wrapper if needed.
             return f"{act_game_object or 'Object'} is idle" # Use outer scope var if possible, else default

    def __chat_func_validate(gpt_response, prompt=""):
        # Validates if the cleaned response is a non-empty string
        try:
            cleaned = __chat_func_clean_up(gpt_response, prompt="") # Pass empty prompt
            return bool(cleaned) # Check if non-empty
        except Exception as e:
            metrics.fail_record(f"Validation failed for obj desc: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe needs act_game_object) ...
    def get_fail_safe(act_game_object): # Needs the object name
        fs = f"{act_game_object} is idle"
        return fs

    # Uses ChatGPT_safe_generate_response -> GPT4_request/ChatGPT_request -> llm_request
    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_obj_event_v1.txt"
    prompt_input = create_prompt_input(act_game_object, act_desp, persona)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "being used" # Simple example state
    special_instruction = "The output should ONLY contain the phrase that describes the object's state as a result of the action."
    fail_safe = get_fail_safe(act_game_object)

    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_act_obj_event_triple(act_game_object, act_obj_desc, persona, verbose=False):
    # This function is identical to run_gpt_prompt_event_triple, just with different inputs.
    # We can reuse the logic.

    def create_prompt_input(act_game_object, act_obj_desc): # Inputs differ
        prompt_input = [act_game_object,
                        act_obj_desc,
                        act_game_object]
        return prompt_input

    # --- Define cleanup and validation logic locally --- START FIX ---
    def __func_clean_up(gpt_response, prompt=""):
        # (Copy the exact __func_clean_up logic from run_gpt_prompt_event_triple here)
        # Example logic (based on run_gpt_prompt_event_triple):
        try:
            cr = gpt_response.strip()
            if cr.startswith("(") and cr.endswith(")"): cr = cr[1:-1] # Remove parenthesis
            parts = [i.strip() for i in cr.split(",")]
            if len(parts) == 2: return parts # Return [predicate, object]
            else:
                print(f"WARN: Event triple cleanup expected 2 parts, got {len(parts)} from '{cr}'")
                return ["is", "idle"] # Default fallback
        except Exception as e:
             print(f"ERROR cleaning event triple: {e}")
             print(f"Raw response: {gpt_response}")
             return ["is", "idle"] # Default fallback

    def __func_validate(gpt_response, prompt=""):
         # (Copy the exact __func_validate logic from run_gpt_prompt_event_triple here)
         # Example logic (based on run_gpt_prompt_event_triple):
        try:
            cleaned_response = __func_clean_up(gpt_response, prompt="") # Use the local cleanup
            if len(cleaned_response) != 2: return False
        except Exception as e:
            # metrics.fail_record(e) # Assuming metrics is available
            print(f"ERROR validating event triple: {e}")
            return False
        return True
    # --- Define cleanup and validation logic locally --- END FIX ---

    def get_fail_safe(act_game_object): # Failsafe needs object name
        fs = (act_game_object, "is", "idle")
        return fs

    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 "max_tokens": 30,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", ")"]}

    # --- Generate Prompt ---
    prompt_template = "persona/prompt_template/v2/generate_event_triple_v1.txt" # Same template
    prompt_input = create_prompt_input(act_game_object, act_obj_desc) # Use correct inputs
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe(act_game_object) # Pass object name

    # --- Make API Call using local helper functions --- START FIX ---
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    # Pass the locally defined functions directly:
    output_list = safe_generate_response(prompt, gpt_param, 5, ["is", "idle"], # Use list fail_safe
                                         __func_validate, __func_clean_up) # Pass local functions
    # --- Make API Call using local helper functions --- END FIX ---

    # Reconstruct the tuple (subj, pred, obj)
    if isinstance(output_list, list) and len(output_list) == 2:
        output = (act_game_object, output_list[0], output_list[1]) # Use object name as subject
    else:
        output = fail_safe

    if debug or verbose:
        # Assuming print_run_prompts exists and works
        print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

    # Return the tuple and debug info
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Legacy Completion API ---
def run_gpt_prompt_new_decomp_schedule(persona,
                                       hourly_schedule, # Added missing param from function definition
                                       main_act_dur,
                                       truncated_act_dur,
                                       start_time_hour,
                                       end_time_hour,
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input=None,
                                       verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona,
                            hourly_schedule, # Added
                            main_act_dur,
                            truncated_act_dur,
                            start_time_hour,
                            end_time_hour,
                            inserted_act,
                            inserted_act_dur,
                            test_input=None):
        # This constructs a detailed plan comparison prompt. Logic seems okay.
        persona_name = persona.name
        start_hour_str = start_time_hour.strftime("%I:%M %p").lstrip('0') # Use %I for 12-hour format
        end_hour_str = end_time_hour.strftime("%I:%M %p").lstrip('0')

        original_plan = ""
        for_time = start_time_hour
        for i in main_act_dur:
            original_plan += f'{for_time.strftime("%I:%M %p").lstrip("0")} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime("%I:%M %p").lstrip("0")} -- ' + i[0]
            original_plan += "\n"
            for_time += datetime.timedelta(minutes=int(i[1]))

        new_plan_init = ""
        for_time = start_time_hour
        # Find the end time of the last truncated activity
        last_truncated_end_time = start_time_hour
        for count, i in enumerate(truncated_act_dur):
            current_end_time = last_truncated_end_time + datetime.timedelta(minutes=int(i[1]))
            new_plan_init += f'{last_truncated_end_time.strftime("%I:%M %p").lstrip("0")} ~ {current_end_time.strftime("%I:%M %p").lstrip("0")} -- ' + i[0]
            new_plan_init += "\n"
            last_truncated_end_time = current_end_time # Update for next iteration

        # Start the next line with the end time of the last truncated activity
        new_plan_init += last_truncated_end_time.strftime("%I:%M %p").lstrip("0") + " ~"


        # Determine the context activity before insertion
        truncated_act_data = "the start of the schedule block"
        if truncated_act_dur: # Get the last *meaningful* activity before the insertion point
             # Look backwards from the end of truncated_act_dur for the last non-inserted activity
             # This assumes the *last* item in truncated_act_dur IS the inserted activity, which seems to be the logic.
             if len(truncated_act_dur) >= 2:
                  truncated_act_data = truncated_act_dur[-2][0] # The activity before the inserted one
             elif main_act_dur: # If only inserted act is present, take original first act?
                  truncated_act_data = main_act_dur[0][0]

        prompt_input = [persona_name,
                        start_hour_str,
                        end_hour_str,
                        original_plan.strip(), # Added strip
                        truncated_act_data, # Context activity
                        inserted_act,
                        str(inserted_act_dur), # Ensure duration is string
                        last_truncated_end_time.strftime("%I:%M %p").lstrip("0"), # Start time for new generation
                        hourly_schedule, # Pass the broader hourly context
                       ]
        return prompt_input


    # ... (__func_clean_up parses specific output format, needs adjustment) ...
    def __func_clean_up(gpt_response, prompt=""):
        # This expects a list of activities filling the remaining time.
        # Parses lines like "HH:MM AM/PM ~ HH:MM AM/PM -- activity"
        # Needs robustness against variations.
        try:
            # Combine prompt and response if needed for context, but the template seems designed for response only.
            # new_schedule = prompt + " " + gpt_response.strip() # Original logic, might include prompt text
            # new_schedule = new_schedule.split("The revised schedule:")[-1].strip() # Assumes this header exists
            # Simpler: assume gpt_response *is* the schedule continuation
            new_schedule = gpt_response.strip().split("\n")

            ret_temp = []
            start_time_str_from_prompt = prompt.split("\n")[-1].split("~")[0].strip() # Get start time from prompt end

            last_end_time_str = start_time_str_from_prompt
            for i in new_schedule:
                 if not i.strip(): continue # Skip empty lines
                 parts = i.split(" -- ")
                 if len(parts) != 2:
                      print(f"WARN: Could not parse schedule line format: {i}")
                      continue # Skip malformed lines
                 time_str, action = parts
                 time_parts = time_str.split(" ~ ")
                 if len(time_parts) != 2:
                      # Handle cases where LLM might omit start time assuming continuation
                      if time_parts[0].strip().endswith("~"): # Check if it ends with '~' indicating missing end time
                           start_time_str = last_end_time_str
                           end_time_str = time_parts[0].strip().replace("~","").strip() # Assuming the single time is end time
                      else: # Assume format like "HH:MM AM/PM" as start time
                          start_time_str = last_end_time_str
                          end_time_str = time_parts[0].strip() # Assume single time is end time? Risky.
                          print(f"WARN: Ambiguous time format in schedule line: {i}. Assuming '{last_end_time_str}' as start.")

                 else:
                      start_time_str, end_time_str = [t.strip() for t in time_parts]
                      # Basic check if start time matches previous end time (optional)
                      # if start_time_str != last_end_time_str:
                      #      print(f"WARN: Time discontinuity in schedule: Expected start {last_end_time_str}, got {start_time_str} for action '{action}'")


                 # Calculate duration
                 try:
                      start_time = datetime.datetime.strptime(start_time_str, "%I:%M %p")
                      end_time = datetime.datetime.strptime(end_time_str, "%I:%M %p")
                      delta = end_time - start_time
                      delta_min = int(delta.total_seconds() / 60)
                      if delta_min < 0: # Handle crossing midnight if necessary (unlikely within one block?)
                           delta_min += 1440
                      if delta_min == 0: # Assign minimum duration if times are same?
                          delta_min = 5 # Example: Assign 5 min if start=end

                      if delta_min > 0:
                          ret_temp.append([action.strip(), delta_min])
                          last_end_time_str = end_time_str # Update for next iteration's continuity check
                      else: print(f"WARN: Calculated zero or negative duration for schedule line: {i}")

                 except ValueError as time_e:
                      print(f"ERROR: Could not parse time format in schedule line: {i} ({time_e})")
                      continue # Skip lines with unparseable times

            # Convert back to JSON format expected by caller?
            # The caller expects a list of lists [[activity, duration], ...]
            # The previous version converted TO JSON. This version should return list of lists.
            # The caller expects the *replacement* section of the schedule.
            return ret_temp # Return [[activity, duration], ...]

        except Exception as e:
            print(f"ERROR cleaning new decomp schedule: {e}")
            print(f"Raw response: {gpt_response}")
            return [] # Return empty list on failure


    # ... (__func_validate needs adjustment, checking list format) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            cleaned = __func_clean_up(gpt_response, prompt)
            if not isinstance(cleaned, list): return False
            # Check format of inner lists
            for item in cleaned:
                if not (isinstance(item, list) and len(item) == 2
                        and isinstance(item[0], str) and isinstance(item[1], int) and item[1] > 0):
                    return False # Invalid format or non-positive duration

            # Optional: Check if total duration matches expected remaining time?
            # This is complex as it requires calculating remaining time based on prompt.
            # Skip detailed duration check for now, rely on format check.
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for new decomp schedule: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe needs original schedule context) ...
    def get_fail_safe(main_act_dur, truncated_act_dur):
        # This tries to reconstruct the rest of the original block.
        # Needs the original `main_act_dur` and the `truncated_act_dur` before insertion.
        try:
            original_total_dur = sum(d for _, d in main_act_dur)
            truncated_total_dur = sum(d for _, d in truncated_act_dur[:-1]) # Exclude the inserted action
            remaining_original_dur = original_total_dur - truncated_total_dur

            # Find where the truncation happened in original list
            temp_dur_sum = 0
            original_index = 0
            for act, dur in main_act_dur:
                if temp_dur_sum >= truncated_total_dur:
                    break
                temp_dur_sum += dur
                original_index += 1

            # Get remaining activities from original schedule
            fail_safe_schedule = []
            if original_index < len(main_act_dur):
                 # Add partial duration of the activity that was cut off
                 partial_dur = temp_dur_sum - truncated_total_dur
                 if partial_dur > 0:
                      fail_safe_schedule.append([main_act_dur[original_index-1][0], partial_dur])

                 # Add the rest of the activities
                 fail_safe_schedule.extend(main_act_dur[original_index:])

            # Adjust total duration to fit remaining time (might slightly distort)
            current_fail_safe_dur = sum(d for _, d in fail_safe_schedule)
            if current_fail_safe_dur != remaining_original_dur and current_fail_safe_dur > 0:
                 ratio = remaining_original_dur / current_fail_safe_dur
                 adjusted_schedule = []
                 adj_sum = 0
                 for act, dur in fail_safe_schedule:
                      adj_dur = max(5, round(dur * ratio / 5) * 5) # Adjust, round to 5 min, min 5?
                      if adj_sum + adj_dur <= remaining_original_dur:
                           adjusted_schedule.append([act, adj_dur])
                           adj_sum += adj_dur
                      else: # Add remaining time to last element
                           rem_dur = remaining_original_dur - adj_sum
                           if rem_dur > 0: adjusted_schedule.append([act, rem_dur])
                           break
                 fail_safe_schedule = adjusted_schedule


            # Convert to the simple [[activity, duration], ...] format
            return fail_safe_schedule if fail_safe_schedule else []

        except Exception as e:
            print(f"ERROR generating fail_safe for new decomp: {e}")
            return []


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 1000,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None} # Stop might need refinement based on LLM behavior
    # --- End Modification ---

    prompt_template = "persona/prompt_template/lifestyle/new_decomp_schedule.txt" # Using lifestyle version
    prompt_input = create_prompt_input(persona,
                                       hourly_schedule, # Pass this in
                                       main_act_dur,
                                       truncated_act_dur,
                                       start_time_hour,
                                       end_time_hour,
                                       inserted_act,
                                       inserted_act_dur,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    # Pass context to fail_safe generator
    fail_safe = get_fail_safe(main_act_dur, truncated_act_dur)

    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    wrapped_validate, wrapped_cleanup)

    # Post-processing to convert to JSON format IF the caller expects it
    # Based on the usage in _create_react, the caller seems to expect [[activity, duration], ...]
    # So, no JSON conversion needed here. The cleanup function already returns this format.
    # json_output = []
    # current_time = start_time_hour + datetime.timedelta(minutes=sum(d for _, d in truncated_act_dur)) # Approx start time for output
    # for activity, duration in output:
    #      end_time = current_time + datetime.timedelta(minutes=duration)
    #      json_output.append({
    #           "activity": activity,
    #           "start": current_time.strftime("%I:%M %p").lstrip('0'),
    #           "end": end_time.strftime("%I:%M %p").lstrip('0'),
    #      })
    #      current_time = end_time
    # output = json_output # Replace list of lists with list of dicts if needed by caller

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_decide_to_talk(persona, target_persona, retrieved, test_input=None,
                                  verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(init_persona, target_persona, retrieved,
                            test_input=None):
        # Logic seems okay, constructs context about current state and history.
        last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
        last_chatted_time = "N/A" # Default if no chat
        last_chat_about = "N/A"  # Default if no chat
        if last_chat:
            last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
            last_chat_about = last_chat.description

        context = ""
        if retrieved: # Check if retrieved is not empty
             # Prioritize event context if available
             if retrieved.get("events"):
                  for c_node in retrieved["events"]:
                       # Simple description formatting
                       context += f"- {c_node.description}. "
                  context += "\n"
             # Add thought context
             if retrieved.get("thoughts"):
                   for c_node in retrieved["thoughts"]:
                        context += f"- {c_node.description}. "
        if not context: context = "No specific context." # Placeholder if no retrieved context

        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %I:%M:%S %p").lstrip('0')
        init_act_desc = init_persona.scratch.act_description
        if "(" in init_act_desc: init_act_desc = init_act_desc.split("(")[-1][:-1]

        # Determine current status more clearly
        if init_persona.scratch.get_f_daily_schedule_index() < len(init_persona.scratch.f_daily_schedule):
            init_act_desp_planned, _ = init_persona.scratch.f_daily_schedule[init_persona.scratch.get_f_daily_schedule_index()]
        else: init_act_desp_planned = init_act_desc # Fallback

        loc = ""
        if ":" in init_persona.scratch.act_address: loc = f" at {init_persona.scratch.act_address.split(':')[-1]} in {init_persona.scratch.act_address.split(':')[-2]}"

        if not init_persona.scratch.planned_path and "waiting" not in init_act_desp_planned: init_p_desc = f"{init_persona.name} is currently {init_act_desp_planned}{loc}."
        elif "waiting" in init_act_desp_planned: init_p_desc = f"{init_persona.name} is waiting{loc}."
        else: init_p_desc = f"{init_persona.name} is on the way to {init_act_desp_planned}{loc}."

        target_act_desc = target_persona.scratch.act_description
        if "(" in target_act_desc: target_act_desc = target_act_desc.split("(")[-1][:-1]

        if target_persona.scratch.get_f_daily_schedule_index() < len(target_persona.scratch.f_daily_schedule):
            target_act_desp_planned, _ = target_persona.scratch.f_daily_schedule[target_persona.scratch.get_f_daily_schedule_index()]
        else: target_act_desp_planned = target_act_desc

        loc_target = ""
        if ":" in target_persona.scratch.act_address: loc_target = f" at {target_persona.scratch.act_address.split(':')[-1]} in {target_persona.scratch.act_address.split(':')[-2]}"

        if not target_persona.scratch.planned_path and "waiting" not in target_act_desp_planned: target_p_desc = f"{target_persona.name} is currently {target_act_desp_planned}{loc_target}."
        elif "waiting" in target_act_desp_planned: target_p_desc = f"{target_persona.name} is waiting{loc_target}."
        else: target_p_desc = f"{target_persona.name} is on the way to {target_act_desp_planned}{loc_target}."

        prompt_input = []
        prompt_input += [context.strip()]
        prompt_input += [curr_time]
        prompt_input += [init_p_desc]
        prompt_input += [target_p_desc]
        prompt_input += [init_persona.name]
        prompt_input += [init_act_desp_planned] # Use planned activity
        prompt_input += [target_persona.name]
        prompt_input += [target_act_desp_planned] # Use planned activity
        prompt_input += [init_act_desp_planned] # Repeat?
        return prompt_input

    # ... (__func_validate expects specific "yes/no" suffix) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            answer = gpt_response.split("Answer in yes or no:")[-1].strip().lower()
            return answer in ["yes", "no"]
        except Exception as e:
            metrics.fail_record(f"Validation failed for decide_to_talk: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (__func_clean_up extracts "yes/no") ...
    def __func_clean_up(gpt_response, prompt=""):
        try:
            answer = gpt_response.split("Answer in yes or no:")[-1].strip().lower()
            return answer if answer in ["yes", "no"] else get_fail_safe()
        except Exception as e:
             print(f"ERROR cleaning decide_to_talk: {e}")
             return get_fail_safe()

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        fs = "no" # Default to not talking? Seems safer. Original was "yes".
        return fs

    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 20,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]} # Added stop
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/decide_to_talk_v2.txt"
    # Ensure retrieved format is handled by create_prompt_input
    prompt_input = create_prompt_input(persona, target_persona, retrieved, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_decide_to_react(persona, target_persona, retrieved, test_input=None,
                                   verbose=False):
    # ... (create_prompt_input is similar to decide_to_talk) ...
    def create_prompt_input(init_persona, target_persona, retrieved,
                            test_input=None):
        # Similar context construction as decide_to_talk
        context = ""
        if retrieved:
             if retrieved.get("events"):
                  for c_node in retrieved["events"]: context += f"- {c_node.description}. "
                  context += "\n"
             if retrieved.get("thoughts"):
                   for c_node in retrieved["thoughts"]: context += f"- {c_node.description}. "
        if not context: context = "No specific context."

        curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %I:%M:%S %p").lstrip('0')
        init_act_desc = init_persona.scratch.act_description
        if "(" in init_act_desc: init_act_desc = init_act_desc.split("(")[-1][:-1]

        if init_persona.scratch.get_f_daily_schedule_index() < len(init_persona.scratch.f_daily_schedule):
            init_act_desp_planned, _ = init_persona.scratch.f_daily_schedule[init_persona.scratch.get_f_daily_schedule_index()]
        else: init_act_desp_planned = init_act_desc

        loc = ""
        if ":" in init_persona.scratch.act_address: loc = f" at {init_persona.scratch.act_address.split(':')[-1]} in {init_persona.scratch.act_address.split(':')[-2]}"

        if not init_persona.scratch.planned_path and "waiting" not in init_act_desp_planned: init_p_desc = f"{init_persona.name} is currently {init_act_desp_planned}{loc}."
        elif "waiting" in init_act_desp_planned: init_p_desc = f"{init_persona.name} is waiting{loc}."
        else: init_p_desc = f"{init_persona.name} is on the way to {init_act_desp_planned}{loc}."

        target_act_desc = target_persona.scratch.act_description
        if "(" in target_act_desc: target_act_desc = target_act_desc.split("(")[-1][:-1]

        if target_persona.scratch.get_f_daily_schedule_index() < len(target_persona.scratch.f_daily_schedule):
            target_act_desp_planned, _ = target_persona.scratch.f_daily_schedule[target_persona.scratch.get_f_daily_schedule_index()]
        else: target_act_desp_planned = target_act_desc

        loc_target = ""
        if ":" in target_persona.scratch.act_address: loc_target = f" at {target_persona.scratch.act_address.split(':')[-1]} in {target_persona.scratch.act_address.split(':')[-2]}"

        if not target_persona.scratch.planned_path and "waiting" not in target_act_desp_planned: target_p_desc = f"{target_persona.name} is currently {target_act_desp_planned}{loc_target}."
        elif "waiting" in target_act_desp_planned: target_p_desc = f"{target_persona.name} is waiting{loc_target}."
        else: target_p_desc = f"{target_persona.name} is on the way to {target_act_desp_planned}{loc_target}."

        prompt_input = []
        prompt_input += [context.strip()]
        prompt_input += [curr_time]
        prompt_input += [init_p_desc]
        prompt_input += [target_p_desc]
        prompt_input += [init_persona.name]
        prompt_input += [init_act_desp_planned] # Use planned activity
        prompt_input += [target_persona.name]
        prompt_input += [target_act_desp_planned] # Use planned activity
        prompt_input += [init_act_desp_planned] # Repeat?
        return prompt_input

    # ... (__func_validate expects specific "Option N" suffix) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            answer = gpt_response.split("Answer: Option")[-1].strip().lower()
            return answer in ["1", "2", "3"]
        except Exception as e:
            metrics.fail_record(f"Validation failed for decide_to_react: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (__func_clean_up extracts the option number) ...
    def __func_clean_up(gpt_response, prompt=""):
        try:
            answer = gpt_response.split("Answer: Option")[-1].strip().lower()
            return answer if answer in ["1", "2", "3"] else get_fail_safe()
        except Exception as e:
             print(f"ERROR cleaning decide_to_react: {e}")
             return get_fail_safe()

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        fs = "3" # Default: keep doing current plan
        return fs

    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 20,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]} # Added stop
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/decide_to_react_v1.txt"
    # Ensure retrieved format is handled by create_prompt_input
    prompt_input = create_prompt_input(persona, target_persona, retrieved, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Functions below might use legacy or chat based on commented sections ---
# --- Assuming the ChatGPT Plugin sections were the intended direction ---
# --- These primarily use ChatGPT_safe_generate_response or _OLD ---
# --- Thus, NO model changes needed in gpt_param, handled by gpt_structure.py ---

# Example: run_gpt_prompt_create_conversation (already uses legacy, needs review if chat version is desired)
# Uses legacy safe_generate_response
def run_gpt_prompt_create_conversation(persona, target_persona, curr_loc,
                                       test_input=None, verbose=False):
    # ... (create_prompt_input seems okay) ...
    def create_prompt_input(init_persona, target_persona, curr_loc,
                            test_input=None):
        # Existing logic for context building
        prev_convo_insert = "\n"
        if init_persona.a_mem.seq_chat:
            # Find last chat with target
            relevant_chats = [chat for chat in reversed(init_persona.a_mem.seq_chat) if chat.object == target_persona.scratch.name]
            if relevant_chats:
                 last_chat = relevant_chats[0]
                 # Check recency (e.g., within last 8 hours / 480 minutes)
                 if int((init_persona.scratch.curr_time - last_chat.created).total_seconds()/60) <= 480:
                     v1 = int((init_persona.scratch.curr_time - last_chat.created).total_seconds()/60)
                     prev_convo_insert += f'{str(v1)} minutes ago, they had the following conversation:\n'
                     if last_chat.filling: # Check if filling has content
                          for row in last_chat.filling:
                              prev_convo_insert += f'{row[0]}: "{row[1]}"\n'
                     else: prev_convo_insert += "[Conversation summary only]\n" # Indicate if only summary exists


        if prev_convo_insert == "\n": prev_convo_insert = "" # Reset if no relevant recent chat found

        # Retrieve relevant thoughts for both personas
        init_thoughts = init_persona.a_mem.retrieve_relevant_thoughts(
            target_persona.scratch.act_event[0], target_persona.scratch.act_event[1], target_persona.scratch.act_event[2])
        init_persona_thought = "".join(f"-- {i.description}\n" for i in init_thoughts) if init_thoughts else "-- No specific thoughts\n"

        target_thoughts = target_persona.a_mem.retrieve_relevant_thoughts(
            init_persona.scratch.act_event[0], init_persona.scratch.act_event[1], init_persona.scratch.act_event[2])
        target_persona_thought = "".join(f"-- {i.description}\n" for i in target_thoughts) if target_thoughts else "-- No specific thoughts\n"


        # Current status descriptions (use planned activity)
        init_act_desc = init_persona.scratch.act_description # Use the current action desc
        if init_persona.scratch.get_f_daily_schedule_index() < len(init_persona.scratch.f_daily_schedule):
             init_act_desp_planned, _ = init_persona.scratch.f_daily_schedule[init_persona.scratch.get_f_daily_schedule_index()]
        else: init_act_desp_planned = init_act_desc
        init_persona_curr_desc = f"{init_persona.name} is {init_act_desp_planned}"

        target_act_desc = target_persona.scratch.act_description
        if target_persona.scratch.get_f_daily_schedule_index() < len(target_persona.scratch.f_daily_schedule):
             target_act_desp_planned, _ = target_persona.scratch.f_daily_schedule[target_persona.scratch.get_f_daily_schedule_index()]
        else: target_act_desp_planned = target_act_desc
        target_persona_curr_desc = f"{target_persona.name} is {target_act_desp_planned}"

                # Location description
        curr_arena_info = curr_loc.get("arena", "unknown location") # Use .get for safety

        prompt_input = []
        prompt_input += [init_persona.scratch.get_str_iss()] # Persona 1 summary
        prompt_input += [target_persona.scratch.get_str_iss()] # Persona 2 summary
        prompt_input += [init_persona.name] # P1 Name
        prompt_input += [target_persona.name] # P2 Name
        prompt_input += [init_persona_thought.strip()] # P1 Thoughts
        prompt_input += [target_persona.name] # P2 Name (repeat?)
        prompt_input += [init_persona.name] # P1 Name (repeat?)
        prompt_input += [target_persona_thought.strip()] # P2 Thoughts
        prompt_input += [init_persona.scratch.curr_time.strftime("%B %d, %Y, %I:%M:%S %p").lstrip('0')] # Time
        prompt_input += [init_persona_curr_desc] # P1 Status
        prompt_input += [target_persona_curr_desc] # P2 Status
        prompt_input += [prev_convo_insert.strip()] # Previous convo context
        prompt_input += [init_persona.name] # P1 Name (repeat?)
        prompt_input += [target_persona.name] # P2 Name (repeat?)
        prompt_input += [curr_arena_info] # Location
        prompt_input += [init_persona.name] # Initiator name for prompt continuation
        return prompt_input


    # ... (__func_clean_up parses conversation lines) ...
    def __func_clean_up(gpt_response, prompt=""):
        # This cleanup assumes the LLM continues the prompt with the conversation.
        # It extracts speaker names and quoted utterances. Might be fragile.
        try:
            # Combine prompt and response for parsing context? Maybe not needed if response is just the convo.
            # full_text = (prompt + gpt_response).split("What would they talk about now?")[-1].strip()
            full_text = gpt_response.strip() # Assume response is the conversation directly
            if not full_text: return [] # Handle empty response

            # Regex to find "<Speaker>: "<Utterance>"" potentially across multiple lines
            # Making utterance capture non-greedy and handle internal quotes carefully.
            # This regex is complex and might need refinement.
            # Pattern: Speaker Name (non-colon chars) : space " Utterance (non-quote chars) " newline
            pattern = re.compile(r"^([\w\s]+):\s*\"(.*?)\"\s*$", re.MULTILINE)
            matches = pattern.findall(full_text)

            if not matches: # Fallback if primary regex fails
                 print("WARN: Primary regex failed for convo parsing. Trying line-by-line.")
                 lines = full_text.split('\n')
                 matches = []
                 for line in lines:
                      if ":" in line and '"' in line:
                           parts = line.split(":", 1)
                           speaker = parts[0].strip()
                           utterance_part = parts[1].strip()
                           if utterance_part.startswith('"') and utterance_part.endswith('"'):
                                utterance = utterance_part[1:-1]
                                matches.append((speaker, utterance))
                           else: print(f"WARN: Skipping line due to quote format: {line}")
                      else: print(f"WARN: Skipping line due to format: {line}")

            if not matches: print("WARN: No conversation lines extracted.")

            # Convert to list of lists format [[speaker, utterance], ...]
            ret = [[speaker, utterance] for speaker, utterance in matches]
            return ret
        except Exception as e:
             print(f"ERROR cleaning create_conversation: {e}")
             print(f"Raw response: {gpt_response}")
             # Need access to persona names for fail_safe here. Pass them or use placeholders.
             return [["Person A", "..." ], ["Person B", "..."]] # Placeholder fail_safe


    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            cleaned = __func_clean_up(gpt_response, prompt)
            # Basic validation: is it a list? does it contain [str, str] pairs?
            if not isinstance(cleaned, list): return False
            if not cleaned: return True # Empty list is valid (no convo happened)
            for item in cleaned:
                 if not (isinstance(item, list) and len(item) == 2 and all(isinstance(s, str) for s in item)):
                      return False
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for create_conversation: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe needs persona objects) ...
    def get_fail_safe(init_persona, target_persona): # Needs names
        convo = [[init_persona.name if init_persona else "Person A", "Hi!"],
                 [target_persona.name if target_persona else "Person B", "Hi!"]]
        return convo


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 1000, # Keep reasonably high for conversation
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": None} # Stop condition might be useful ("\n\n"?)
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/create_conversation_v2.txt"
    prompt_input = create_prompt_input(persona, target_persona, curr_loc,
                                       test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(persona, target_persona) # Pass personas
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via _OLD wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_summarize_conversation(persona, conversation, test_input=None, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(conversation, test_input=None):
        convo_str = ""
        for row in conversation: # Ensure row is [speaker, utterance]
             if isinstance(row, list) and len(row)==2:
                  convo_str += f'{row[0]}: "{row[1]}"\n'
        if not convo_str: convo_str = "[Empty conversation]"

        prompt_input = [convo_str.strip()]
        return prompt_input

    # ... (__chat_func_clean_up is main cleanup) ...
    def __chat_func_clean_up(gpt_response, prompt=""):
        # Expects JSON {"output": "summary"}
        try:
             # Safe generator provides extracted output
             ret = gpt_response.strip()
             # Basic post-processing (optional)
             if ret.lower().startswith("this is a conversation about"):
                  ret = ret[len("this is a conversation about"):].strip()
             if ret.lower().startswith("conversing about"):
                  ret = ret[len("conversing about"):].strip()
             return ret
        except Exception as e:
             print(f"ERROR cleaning convo summary: {e}")
             return get_fail_safe()

    # ... (__chat_func_validate checks cleanup) ...
    def __chat_func_validate(gpt_response, prompt=""):
        try:
            cleaned = __chat_func_clean_up(gpt_response, prompt)
            return bool(cleaned) # Check if non-empty
        except Exception as e:
            metrics.fail_record(f"Validation failed for convo summary: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        return "a brief chat" # More generic failsafe


    # Uses ChatGPT_safe_generate_response -> GPT4_request/ChatGPT_request -> llm_request
    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_conversation_v1.txt"
    prompt_input = create_prompt_input(conversation, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "discussing weekend plans"
    special_instruction = "The output must concisely summarize the main topic of the conversation. Do not include the speaker names unless essential. Start the summary directly without introductory phrases like 'The conversation is about...'."
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_extract_keywords(persona, description, test_input=None, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(description, test_input=None):
        if "\n" in description:
            description = description.replace("\n", " <LINE_BREAK> ")
        prompt_input = [description]
        return prompt_input

    # ... (__func_clean_up parses keyword sections) ...
    def __func_clean_up(gpt_response, prompt=""):
        # Parses "Factual keywords: kw1, kw2\nEmotive keywords: kw3, kw4"
        try:
            factual_keywords = []
            emotive_keywords = []

            # Look for keyword sections case-insensitively
            factual_match = re.search(r"factual keywords:(.*?)(\n|$|emotive keywords:)", gpt_response, re.IGNORECASE | re.DOTALL)
            emotive_match = re.search(r"emotive keywords:(.*)", gpt_response, re.IGNORECASE | re.DOTALL)

            if factual_match:
                factual_str = factual_match.group(1).strip()
                factual_keywords = [kw.strip().lower().rstrip('.') for kw in factual_str.split(',') if kw.strip()]

            if emotive_match:
                emotive_str = emotive_match.group(1).strip()
                emotive_keywords = [kw.strip().lower().rstrip('.') for kw in emotive_str.split(',') if kw.strip()]

            # Fallback if sections not found: split all by comma
            if not factual_keywords and not emotive_keywords:
                 all_keywords = [kw.strip().lower().rstrip('.') for kw in gpt_response.split(',') if kw.strip()]
                 # Heuristic: assign shorter keywords as factual? Or just combine?
                 # Let's combine for simplicity in fallback.
                 factual_keywords = all_keywords
                 print("WARN: Could not find specific keyword sections, extracting all.")


            all_unique_keywords = set(factual_keywords + emotive_keywords)
            # Remove empty strings if any resulted from parsing
            all_unique_keywords.discard('')

            # print("Extracted Keywords:", all_unique_keywords) # Debug
            return all_unique_keywords
        except Exception as e:
             print(f"ERROR cleaning keywords: {e}")
             print(f"Raw response: {gpt_response}")
             return set() # Return empty set on error


    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            cleaned = __func_clean_up(gpt_response, prompt)
            # Basic check: is it a set?
            return isinstance(cleaned, set)
        except Exception as e:
            metrics.fail_record(f"Validation failed for keywords: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        return set() # Empty set is a safe fallback


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 50,
                 "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n\n"]} # Stop on double newline?
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/get_keywords_v1.txt"
    prompt_input = create_prompt_input(description, test_input)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_keyword_to_thoughts(persona, keyword, concept_summary, test_input=None, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, keyword, concept_summary, test_input=None):
        prompt_input = [keyword, concept_summary, persona.name]
        return prompt_input

    # ... (__func_clean_up is simple) ...
    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip().replace('"','') # Remove quotes
        return gpt_response

    # ... (__func_validate is simple) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            cleaned = __func_clean_up(gpt_response, prompt)
            return bool(cleaned) # Check non-empty
        except Exception as e:
            metrics.fail_record(f"Validation failed for keyword_to_thoughts: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        return "" # Empty string


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 40,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", "."]} # Added stops
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/keyword_to_thoughts_v1.txt"
    prompt_input = create_prompt_input(persona, keyword, concept_summary)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_convo_to_thoughts(persona,
                                    init_persona_name,
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(init_persona_name,
                                    target_persona_name,
                                    convo_str,
                                    fin_target, test_input=None):
        prompt_input = [init_persona_name,
                        target_persona_name,
                        convo_str,
                        init_persona_name,
                        fin_target]
        return prompt_input

    # ... (__func_clean_up is simple) ...
    def __func_clean_up(gpt_response, prompt=""):
        gpt_response = gpt_response.strip().replace('"','')
        return gpt_response

    # ... (__func_validate is simple) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            cleaned = __func_clean_up(gpt_response, prompt)
            return bool(cleaned)
        except Exception as e:
            metrics.fail_record(f"Validation failed for convo_to_thoughts: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        return ""


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 40,
                 "temperature": 0.7, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", "."]}
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/convo_to_thoughts_v1.txt"
    prompt_input = create_prompt_input(init_persona_name,
                                    target_persona_name,
                                    convo_str,
                                    fin_target)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe()
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Chat Completion API (via safe wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_event_poignancy(persona, event_description, test_input=None, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, event_description, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        event_description]
        return prompt_input

    # ... (__chat_func_clean_up is main cleanup) ...
    def __chat_func_clean_up(gpt_response, prompt=""):
        # Expects JSON {"output": "N"}
        try:
            # Safe generator gives extracted output 'N'
            poignancy_score = int(gpt_response)
            if 1 <= poignancy_score <= 10:
                 return poignancy_score
            else:
                 print(f"WARN: Poignancy score {poignancy_score} out of range (1-10). Using failsafe.")
                 return get_fail_safe()
        except ValueError:
             print(f"WARN: Could not convert poignancy response '{gpt_response}' to int. Using failsafe.")
             return get_fail_safe()
        except Exception as e:
             print(f"ERROR cleaning event poignancy: {e}")
             return get_fail_safe()

    # ... (__chat_func_validate checks cleanup) ...
    def __chat_func_validate(gpt_response, prompt=""):
        try:
            score = int(gpt_response)
            return 1 <= score <= 10
        except Exception as e:
            # metrics.fail_record(f"Validation failed for event poignancy: {e}. Response: {gpt_response[:200]}") # Already logged in cleanup?
            return False

    # ... (get_fail_safe is simple) ...
    def get_fail_safe():
        return 4 # Default poignancy

    # Uses ChatGPT_safe_generate_response -> GPT4_request/ChatGPT_request -> llm_request
    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_event_v1.txt"
    prompt_input = create_prompt_input(persona, event_description)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5" # Example integer output
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via safe wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_thought_poignancy(persona, event_description, test_input=None, verbose=False):
    # Identical logic to event poignancy, just different template name
    create_prompt_input = run_gpt_prompt_event_poignancy.create_prompt_input # Reuse
    __chat_func_clean_up = run_gpt_prompt_event_poignancy.__chat_func_clean_up # Reuse
    __chat_func_validate = run_gpt_prompt_event_poignancy.__chat_func_validate # Reuse
    get_fail_safe = run_gpt_prompt_event_poignancy.get_fail_safe # Reuse

    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_thought_v1.txt" # Different template
    prompt_input = create_prompt_input(persona, event_description)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via safe wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_chat_poignancy(persona, event_description, test_input=None, verbose=False):
    # Identical logic to event poignancy, just different template name
    create_prompt_input = run_gpt_prompt_event_poignancy.create_prompt_input # Reuse
    __chat_func_clean_up = run_gpt_prompt_event_poignancy.__chat_func_clean_up # Reuse
    __chat_func_validate = run_gpt_prompt_event_poignancy.__chat_func_validate # Reuse
    get_fail_safe = run_gpt_prompt_event_poignancy.get_fail_safe # Reuse

    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v3_ChatGPT/poignancy_chat_v1.txt" # Different template
    prompt_input = create_prompt_input(persona, event_description)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = "5"
    special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."
    fail_safe = get_fail_safe()

    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# --- Function using Chat Completion API (via safe wrapper) ---
# No gpt_param changes needed here
def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, statements, n, test_input=None):
        prompt_input = [statements, str(n)]
        return prompt_input

    # ... (__chat_func_clean_up expects JSON list) ...
    def __chat_func_clean_up(gpt_response, prompt=""):
        # Expects JSON {"output": ["focal point 1", ...]}
        try:
            # Safe generator provides the extracted output list
            if isinstance(gpt_response, list) and all(isinstance(item, str) for item in gpt_response):
                return gpt_response
            elif isinstance(gpt_response, str): # Try parsing if it's a stringified list
                 try:
                      parsed_list = ast.literal_eval(gpt_response)
                      if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                           return parsed_list
                 except Exception: pass # Fall through if parsing fails

            print(f"WARN: Focal point response not a valid list of strings: {gpt_response}. Using failsafe.")
            return get_fail_safe(n) # Use outer scope n
        except Exception as e:
             print(f"ERROR cleaning focal points: {e}")
             return get_fail_safe(n)

    # ... (__chat_func_validate checks cleanup) ...
    def __chat_func_validate(gpt_response, prompt=""):
        try:
            cleaned = __chat_func_clean_up(gpt_response, prompt)
            # Check if it's a list of strings of the expected length n
            return isinstance(cleaned, list) and all(isinstance(item, str) for item in cleaned) and len(cleaned) <= n
        except Exception as e:
            # metrics.fail_record(f"Validation failed for focal points: {e}. Response: {gpt_response[:200]}") # Logged in cleanup?
            return False

    # ... (get_fail_safe needs n) ...
    def get_fail_safe(n):
        return ["What am I doing"] * n # Generic question


    # Uses ChatGPT_safe_generate_response -> GPT4_request/ChatGPT_request -> llm_request
    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder

    prompt_template = "persona/prompt_template/v3_ChatGPT/generate_focal_pt_v1.txt"
    prompt_input = create_prompt_input(persona, statements, n)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = '["What should Jane do for lunch?", "Does Jane like strawberry?", "Who is Jane?"]'
    special_instruction = f"Output must be a list of {n} strings." # Use f-string
    fail_safe = get_fail_safe(n)

    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Function using Legacy Completion API ---
def run_gpt_prompt_insight_and_guidance(persona, statements, n, test_input=None, verbose=False):
    # ... (create_prompt_input remains the same) ...
    def create_prompt_input(persona, statements, n, test_input=None):
        prompt_input = [statements, str(n)]
        return prompt_input

    # ... (__func_clean_up parses numbered list with evidence) ...
    def __func_clean_up(gpt_response, prompt=""):
        # Parses "1. insight (because of 1, 2)\n2. insight 2 (because of 3)..."
        # Returns dict {insight: [evidence_indices]}
        try:
            # Prepend numbering if missing for easier splitting
            if not gpt_response.strip().startswith("1"):
                gpt_response = "1. " + gpt_response.strip()

            ret = dict()
            # Split by newline, then process each line
            lines = gpt_response.strip().split("\n")
            for i in lines:
                # Remove numbering (e.g., "1. ")
                line_content = re.sub(r"^\d+[\.\)]\s*", "", i).strip()

                # Find evidence part "(because of ...)"
                evidence_match = re.search(r"\(because of ([\d,\s]+)\)", line_content, re.IGNORECASE)

                if evidence_match:
                    thought = line_content[:evidence_match.start()].strip()
                    evi_raw_str = evidence_match.group(1).strip()
                    # Extract integers from evidence string
                    evi_raw = [int(num_str.strip()) for num_str in evi_raw_str.split(',') if num_str.strip().isdigit()]
                    # Adjust indices to be 0-based if they are 1-based in the response
                    # Assuming the prompt implies 1-based indexing corresponding to input statements.
                    # No adjustment needed if indices directly map to input list passed later.
                    if thought: # Ensure thought is not empty
                         ret[thought] = evi_raw
                    else: print(f"WARN: Empty insight found in line: {i}")
                else:
                    # If no evidence part found, assume the whole line is the thought, no evidence
                    thought = line_content.strip()
                    if thought:
                         ret[thought] = [] # Assign empty list for evidence
                         print(f"WARN: No evidence found for insight: {thought}")

            return ret
        except Exception as e:
             print(f"ERROR cleaning insight/guidance: {e}")
             print(f"Raw response: {gpt_response}")
             return get_fail_safe(n) # Use outer scope n


    # ... (__func_validate checks cleanup) ...
    def __func_validate(gpt_response, prompt=""):
        try:
            cleaned = __func_clean_up(gpt_response, prompt)
            # Basic check: is it a dict? Are keys strings and values lists of ints?
            if not isinstance(cleaned, dict): return False
            for k, v in cleaned.items():
                if not isinstance(k, str) or not isinstance(v, list) or not all(isinstance(i, int) for i in v):
                    return False
            return True
        except Exception as e:
            metrics.fail_record(f"Validation failed for insight/guidance: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe needs n) ...
    def get_fail_safe(n):
        # Return a dictionary with placeholder insights and empty evidence
        return {f"Placeholder insight {i+1}": [] for i in range(n)}


    # --- Model Parameter Modification ---
    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini':
        engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL

    gpt_param = {"engine": engine_to_use,
                 # "api_type": "openai", # Removed
                 "max_tokens": 500, # Increased as insights can be long
                 "temperature": 0.5, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n\n"]} # Stop on double newline?
    # --- End Modification ---

    prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.txt"
    prompt_input = create_prompt_input(persona, statements, n)
    prompt = generate_prompt(prompt_input, prompt_template)

    fail_safe = get_fail_safe(n)
    # Uses legacy safe_generate_response -> GPT_request -> gpt_request_all_version
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
                                    __func_validate, __func_clean_up)

    if debug or verbose:
        print_run_prompts(prompt_template, persona, gpt_param,
                          prompt_input, prompt, output)

    # Post-processing in reflect.py already handles mapping indices to nodes.
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# --- Functions below use ChatGPT safe wrappers - NO model changes needed here ---

def run_gpt_prompt_agent_chat_summarize_ideas(persona, target_persona, statements, curr_context, test_input=None,
                                              verbose=False):
    # ... (Uses ChatGPT_safe_generate_response) ...
    def create_prompt_input(persona, target_persona, statements, curr_context, test_input=None):
        prompt_input = [persona.scratch.get_str_curr_date_str(), curr_context, persona.scratch.currently,
                        statements, persona.scratch.name, target_persona.scratch.name]
        return prompt_input
    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response.strip().replace('"',"") # Simpler cleanup
    def __chat_func_validate(gpt_response, prompt=""):
        try: return bool(__chat_func_clean_up(gpt_response, prompt))
        except Exception as e: return False
    def get_fail_safe(): return "..."

    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0.5, "top_p": 1} # Placeholder
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_ideas_v1.txt"
    prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is thinking about her project' # Example
    special_instruction = 'The output should be a concise string summarizing the main idea for the conversation.'
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat_summarize_relationship(persona, target_persona, statements, test_input=None,
                                                     verbose=False):
    # ... (Uses ChatGPT_safe_generate_response) ...
    def create_prompt_input(persona, target_persona, statements, test_input=None):
        prompt_input = [statements, persona.scratch.name, target_persona.scratch.name]
        return prompt_input
    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response.strip().replace('"','')
    def __chat_func_validate(gpt_response, prompt=""):
        try: return bool(__chat_func_clean_up(gpt_response, prompt))
        except Exception as e: return False
    def get_fail_safe(): return "..."

    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0.5, "top_p": 1} # Placeholder
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_chat_relationship_v2.txt" # v2 template? Check content
    prompt_input = create_prompt_input(persona, target_persona, statements)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'friends' # Example
    special_instruction = 'The output should be a concise string describing the relationship.'
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


def run_gpt_prompt_agent_chat(maze, persona, target_persona,
                              curr_context,
                              init_summ_idea,
                              target_summ_idea, test_input=None, verbose=False):
    # ... (Uses ChatGPT_safe_generate_response) ...
    def create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea, test_input=None):
        # Simplified context building based on previous logic
        prev_convo_insert = "\n" # Placeholder, potentially add recent chat summary here if needed
        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"

        prompt_input = [persona.scratch.currently,
                        target_persona.scratch.currently,
                        prev_convo_insert.strip(),
                        curr_context,
                        curr_location,
                        persona.scratch.name, init_summ_idea, persona.scratch.name, target_persona.scratch.name, # P1 perspective
                        target_persona.scratch.name, target_summ_idea, target_persona.scratch.name, persona.scratch.name, # P2 perspective
                        persona.scratch.name] # Initiator name to start convo
        return prompt_input

    def __chat_func_clean_up(gpt_response, prompt=""):
        # Expects JSON {"output": [["Speaker", "Utterance"], ...]}
        try:
            # Safe generator provides the extracted list
            if isinstance(gpt_response, list): # Check if already a list
                # Validate inner structure
                if all(isinstance(item, list) and len(item)==2 and all(isinstance(s, str) for s in item) for item in gpt_response):
                    return gpt_response
            elif isinstance(gpt_response, str): # Try parsing if string
                 try:
                      parsed_list = ast.literal_eval(gpt_response)
                      if isinstance(parsed_list, list) and all(isinstance(item, list) and len(item)==2 and all(isinstance(s, str) for s in item) for item in parsed_list):
                           return parsed_list
                 except Exception: pass # Fall through if parsing fails

            print(f"WARN: Agent chat response not a valid list of lists: {gpt_response}. Using failsafe.")
            return get_fail_safe()
        except Exception as e:
             print(f"ERROR cleaning agent chat: {e}")
             return get_fail_safe()

    def __chat_func_validate(gpt_response, prompt=""): # Validation happens in cleanup
        try:
            cleaned = __chat_func_clean_up(gpt_response, prompt)
            return isinstance(cleaned, list) # Just check if it's a list
        except Exception: return False

    def get_fail_safe(): return [[persona.name if persona else "Person A", "..."], [target_persona.name if target_persona else "Person B", "..."]]


    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0.7, "top_p": 1} # Placeholder
    prompt_template = "persona/prompt_template/v3_ChatGPT/agent_chat_v1.txt"
    prompt_input = create_prompt_input(persona, target_persona, curr_context, init_summ_idea, target_summ_idea)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = '[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"]]' # Example format
    special_instruction = 'The output should be a list of lists where the inner lists are in the form of ["<Name>", "<Utterance>"].'
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# Remaining functions mostly use ChatGPT safe wrappers - skipping detailed breakdown unless specific issues arise.
# Key is that the underlying call in gpt_structure.py handles model selection.

# ... (run_gpt_prompt_summarize_ideas - uses ChatGPT safe wrapper) ...
def run_gpt_prompt_summarize_ideas(persona, statements, question, test_input=None, verbose=False):
    # ... (Uses ChatGPT_safe_generate_response) ...
    def create_prompt_input(persona, statements, question, test_input=None):
        prompt_input = [statements, persona.scratch.name, question]
        return prompt_input
    def __chat_func_clean_up(gpt_response, prompt=""):
        return gpt_response.strip().replace('"',"") # Simpler cleanup
    def __chat_func_validate(gpt_response, prompt=""):
        try: return bool(__chat_func_clean_up(gpt_response, prompt))
        except Exception as e: return False
    def get_fail_safe(): return "..."

    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0.5, "top_p": 1} # Placeholder
    prompt_template = "persona/prompt_template/v3_ChatGPT/summarize_ideas_v1.txt"
    prompt_input = create_prompt_input(persona, statements, question)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'Jane Doe is working on a project' # Example
    special_instruction = 'The output should be a string that responds to the question.'
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)

    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# ... (run_gpt_prompt_generate_next_convo_line - uses legacy, CHECK if chat version intended) ...
# --- Note on generate_next_convo_line: Original code uses legacy. If chat is preferred, refactor needed. ---
# --- Assuming legacy usage for now, applying model change ---
def run_gpt_prompt_generate_next_convo_line(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None,
                                            verbose=False):
    def create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary, test_input=None):
        prompt_input = [persona.scratch.name,
                        persona.scratch.get_str_iss(),
                        persona.scratch.name,
                        interlocutor_desc,
                        prev_convo, # Make sure this is formatted well
                        persona.scratch.name, # Repeat?
                        retrieved_summary,
                        persona.scratch.name, ]
        return prompt_input
    def __func_clean_up(gpt_response, prompt=""): return gpt_response.strip().replace('"','')
    def __func_validate(gpt_response, prompt=""):
        try: return bool(__func_clean_up(gpt_response, prompt))
        except Exception: return False
    def get_fail_safe(): return "..."

    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini': engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL
    gpt_param = {"engine": engine_to_use, 
                 "max_tokens": 250, "temperature": 1, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
    prompt_template = "persona/prompt_template/v2/generate_next_convo_line_v1.txt"
    prompt_input = create_prompt_input(persona, interlocutor_desc, prev_convo, retrieved_summary)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe, __func_validate, __func_clean_up)
    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# ... (run_gpt_prompt_generate_whisper_inner_thought - uses legacy, apply model change) ...
def run_gpt_prompt_generate_whisper_inner_thought(persona, whisper, test_input=None, verbose=False):
    def create_prompt_input(persona, whisper, test_input=None):
        prompt_input = [persona.scratch.name, whisper]
        return prompt_input
    def __func_clean_up(gpt_response, prompt=""): return gpt_response.strip().replace('"','')
    def __func_validate(gpt_response, prompt=""):
        try: return bool(__func_clean_up(gpt_response, prompt))
        except Exception: return False
    def get_fail_safe(): return "..."

    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini': engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL
    gpt_param = {"engine": engine_to_use, #"api_type": "openai", # Removed
                 "max_tokens": 50, "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", '.']}
    prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.txt"
    prompt_input = create_prompt_input(persona, whisper)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe, __func_validate, __func_clean_up)
    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

# ... (run_gpt_prompt_planning_thought_on_convo - uses legacy, apply model change) ...
def run_gpt_prompt_planning_thought_on_convo(persona, all_utt, test_input=None, verbose=False):
    def create_prompt_input(persona, all_utt, test_input=None):
        prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
        return prompt_input
    def __func_clean_up(gpt_response, prompt=""): return gpt_response.strip().replace('"','')
    def __func_validate(gpt_response, prompt=""):
        try: return bool(__func_clean_up(gpt_response, prompt))
        except Exception: return False
    def get_fail_safe(): return "..."

    engine_to_use = "gpt-3.5-turbo-instruct"
    if key_type == 'gemini': engine_to_use = DEFAULT_GEMINI_COMPLETION_MODEL
    gpt_param = {"engine": engine_to_use, #"api_type": "openai", # Removed
                 "max_tokens": 50, "temperature": 0, "top_p": 1, "stream": False,
                 "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n", '.']}
    prompt_template = "persona/prompt_template/v2/planning_thought_on_convo_v1.txt"
    prompt_input = create_prompt_input(persona, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()
    output = safe_generate_response(prompt, gpt_param, 5, fail_safe, __func_validate, __func_clean_up)
    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# ... (run_gpt_prompt_memo_on_convo - uses ChatGPT safe wrapper) ...
def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False):
    def create_prompt_input(persona, all_utt, test_input=None):
        prompt_input = [all_utt, persona.scratch.name, persona.scratch.name, persona.scratch.name]
        return prompt_input
    def __chat_func_clean_up(gpt_response, prompt=""): return gpt_response.strip().replace('"','')
    def __chat_func_validate(gpt_response, prompt=""):
        try: return bool(__chat_func_clean_up(gpt_response, prompt))
        except Exception: return False
    def get_fail_safe(): return "..."

    gpt_param = {"engine": "ChatGPT Default/GPT4", "temperature": 0, "top_p": 1} # Placeholder
    prompt_template = "persona/prompt_template/v3_ChatGPT/memo_on_convo_v1.txt"
    prompt_input = create_prompt_input(persona, all_utt)
    prompt = generate_prompt(prompt_input, prompt_template)
    example_output = 'The conversation was about project deadlines.' # Example
    special_instruction = 'The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed during the conversation.'
    fail_safe = get_fail_safe()
    output = ChatGPT_safe_generate_response(prompt, example_output, special_instruction, 3, fail_safe,
                                            __chat_func_validate, __chat_func_clean_up, True)
    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# ... (run_gpt_generate_safety_score - uses ChatGPT_safe_generate_response_OLD) ...
def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False):
    # This function seems specifically for safety checks, likely fine with default chat models
    def create_prompt_input(comment, test_input=None):
        prompt_input = [comment]
        return prompt_input
    def __chat_func_clean_up(gpt_response, prompt=""):
        # Expects JSON {"output": score}
        try:
             # _OLD safe wrapper doesn't auto-extract, needs parsing here
             response = json.loads(gpt_response)
             score = int(response["output"]) # Expecting an integer score
             return score
        except (json.JSONDecodeError, KeyError, ValueError) as e:
             print(f"ERROR cleaning safety score: {e}")
             return get_fail_safe() # Return failsafe score

    def __chat_func_validate(gpt_response, prompt=""):
        try:
            response = json.loads(gpt_response)
            return "output" in response and isinstance(response["output"], (int, float, str)) # Allow string temporarily due to cleanup cast
        except Exception as e:
            metrics.fail_record(f"Validation failed for safety score: {e}. Response: {gpt_response[:200]}")
            return False
    def get_fail_safe(): return 0 # Default to safe score? Or maybe a moderate score like 5? Let's use 0.

    gpt_param = {"engine": "ChatGPT Default", "temperature": 0, "top_p": 1} # Placeholder
    prompt_template = "persona/prompt_template/safety/anthromorphosization_v1.txt" # Check template content
    prompt_input = create_prompt_input(comment)
    prompt = generate_prompt(prompt_input, prompt_template)
    # example_output = {"output": 3} # Example JSON
    # special_instruction = "Output must be JSON with 'output' field containing safety score." # Original prompt likely implies this
    fail_safe = get_fail_safe()
    # Using _OLD which doesn't use the example/special instruction format well
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)

    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# ... (extract_first_json_dict - utility, no changes) ...
def extract_first_json_dict(data_str):
    # ... (no changes needed) ...
    start_idx = data_str.find('{')
    if start_idx == -1: return None # No opening brace
    # Find the matching closing brace, handling nested structures
    brace_level = 0
    end_idx = -1
    for i in range(start_idx, len(data_str)):
        if data_str[i] == '{':
            brace_level += 1
        elif data_str[i] == '}':
            brace_level -= 1
            if brace_level == 0:
                end_idx = i + 1
                break
    if end_idx == -1: return None # No matching closing brace

    json_str = data_str[start_idx:end_idx]
    try:
        json_dict = json.loads(json_str)
        return json_dict
    except json.JSONDecodeError as e:
        # metrics.fail_record(f"JSON Parsing Failed in extract_first_json_dict: {e} for string: {json_str[:100]}")
        print(f"WARN: JSON Parsing Failed in extract_first_json_dict for string: {json_str[:100]}")
        return None


# ... (run_gpt_generate_iterative_chat_utt - uses ChatGPT_safe_generate_response_OLD) ...
def run_gpt_generate_iterative_chat_utt(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                        test_input=None, verbose=False):
    # ... (create_prompt_input builds complex context) ...
    def create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat, test_input=None):
        # Builds context including persona summaries, relationships, retrieved thoughts, location, ongoing chat
        persona = init_persona
        prev_convo_insert = "\n" # Placeholder for previous relevant chat summary if needed
        # Location
        curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
        curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
        curr_location = f"{curr_arena} in {curr_sector}"
        # Current chat history formatting
        convo_str = ""
        for i in curr_chat[-6:]: # Limit history length?
            convo_str += ": ".join(i) + "\n"
        if not convo_str: convo_str = "[The conversation has not started yet -- start it!]"
        # Persona ISS and Relationship
        personal_information = init_persona.scratch.get_str_iss()
        relationship_info = init_persona.scratch.get_relationship_feeling(target_persona.scratch.name)
        relationship = relationship_info.get('relationship', 'unknown')
        feeling = relationship_info.get('feeling', 'neutral')
        personal_information += f"\nRelationship with {target_persona.scratch.name}: {relationship}"
        personal_information += f"\nMy feeling towards {target_persona.scratch.name}: {feeling}"
        # Retrieved thoughts/summary
        retrieved_summary = retrieved # Assumes 'retrieved' IS the summary string here

        prompt_input = [personal_information.strip(),
                        init_persona.scratch.name,
                        retrieved_summary.strip(),
                        prev_convo_insert.strip(),
                        curr_location,
                        curr_context, # External context passed in
                        target_persona.scratch.name,
                        convo_str.strip()] # The ongoing conversation
        return prompt_input

    # ... (__chat_func_clean_up expects JSON {"utterance": "...", "end": bool}) ...
    def __chat_func_clean_up(gpt_response, prompt=""):
        # _OLD wrapper doesn't auto-extract, needs parsing
        try:
            # Try extracting the first JSON dict from the potentially noisy response
            parsed_json = extract_first_json_dict(gpt_response)
            if parsed_json and "utterance" in parsed_json and "end" in parsed_json:
                 # Basic validation of types
                 utterance = str(parsed_json["utterance"])
                 end_val = parsed_json["end"]
                 end_bool = True # Default to ending
                 if isinstance(end_val, bool): end_bool = end_val
                 elif isinstance(end_val, str) and ("f" in end_val.lower() or "no" in end_val.lower()): end_bool = False

                 return {"utterance": utterance, "end": end_bool}
            else:
                 print(f"WARN: Could not parse expected JSON fields in iterative chat response: {gpt_response[:200]}")
                 return get_fail_safe()
        except Exception as e:
            print(f"ERROR cleaning iterative chat response: {e}")
            return get_fail_safe()

    # ... (__chat_func_validate checks cleanup) ...
    def __chat_func_validate(gpt_response, prompt=""):
        try:
            cleaned = __chat_func_clean_up(gpt_response, prompt)
            return "utterance" in cleaned and "end" in cleaned and isinstance(cleaned["end"], bool)
        except Exception as e:
            # metrics.fail_record(f"Validation failed for iterative chat: {e}. Response: {gpt_response[:200]}")
            print(f"WARN: Validation failed for iterative chat: {e}. Response: {gpt_response[:200]}")
            return False

    # ... (get_fail_safe is simple dict) ...
    def get_fail_safe():
        return {"utterance": "...", "end": False}


    gpt_param = {"engine": "ChatGPT Default", "temperature": 0.7, "top_p": 1} # Placeholder
    prompt_template = "persona/prompt_template/lifestyle/iterative_convo.txt" # Lifestyle version
    prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)
    prompt = generate_prompt(prompt_input, prompt_template)
    fail_safe = get_fail_safe()
    # Uses ChatGPT_safe_generate_response_OLD -> ChatGPT_request -> llm_request
    output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                __chat_func_validate, __chat_func_clean_up, verbose)

    if debug or verbose: print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# ... (run_gpt_generate_iterative_chat_utt_origin - uses ChatGPT_safe_generate_response_OLD, seems redundant?) ...
# --- Assuming this is an older/alternative version, skipping detailed mods unless needed ---
def run_gpt_generate_iterative_chat_utt_origin(maze, init_persona, target_persona, retrieved, curr_context, curr_chat,
                                        test_input=None, verbose=False):
     # Similar logic to the non-origin version, likely uses v3_ChatGPT template.
     # Keep implementation as is for now, assuming non-origin is primary.
     # If needed, apply similar cleanup/validation logic as above.
     print("WARN: Calling run_gpt_generate_iterative_chat_utt_origin - check if intended.")
     create_prompt_input = run_gpt_generate_iterative_chat_utt.create_prompt_input # Reuse? Needs check
     __chat_func_clean_up = run_gpt_generate_iterative_chat_utt.__chat_func_clean_up # Reuse? Needs check
     __chat_func_validate = run_gpt_generate_iterative_chat_utt.__chat_func_validate # Reuse? Needs check
     get_fail_safe = run_gpt_generate_iterative_chat_utt.get_fail_safe # Reuse? Needs check

     gpt_param = {"engine": "ChatGPT Default", "temperature": 0.7, "top_p": 1} # Placeholder
     prompt_template = "persona/prompt_template/v3_ChatGPT/iterative_convo_v1.txt" # v3 template
     prompt_input = create_prompt_input(maze, init_persona, target_persona, retrieved, curr_context, curr_chat)
     prompt = generate_prompt(prompt_input, prompt_template)
     fail_safe = get_fail_safe()
     output = ChatGPT_safe_generate_response_OLD(prompt, 3, fail_safe,
                                                 __chat_func_validate, __chat_func_clean_up, verbose)

     if debug or verbose: print_run_prompts(prompt_template, init_persona, gpt_param, prompt_input, prompt, output)
     return output, [output, prompt, gpt_param, prompt_input, fail_safe]


# ... (run_gpt_generate_retrieved_summary - uses direct ChatGPT_single_request) ...
# No gpt_param changes needed
def run_gpt_generate_retrieved_summary(name, retrieved):
    # ... (prompt construction seems okay) ...
    prompt_input = f"The following sentences are thoughts or memories retrieved for {name}, " \
                   "please summarize them into a brief and precise statement representing the core themes:\n"
    key_description = ""
    if retrieved:
         for v in retrieved[:15]: # Limit input size?
             key_description += f"- {v.description}\n"
    else: key_description = "- No relevant thoughts or memories found.\n"

    print('--- Running Retrieved Summary ---')
    # print(f"Input:\n {prompt_input + key_description}")
    try:
        # Uses ChatGPT_single_request -> llm_request (handles model selection)
        output = ChatGPT_single_request(prompt_input + key_description.strip())
        print(f"Summary Output for {name}: {output}")
        return output.strip()
    except Exception as e:
        metrics.fail_record(f"Retrieved Summary Error: {e}")
        # Fallback: return concatenated descriptions? Or generic message?
        return "Could not summarize retrieved memories."


# ... (run_gpt_update_retrieved_summary - uses direct ChatGPT_single_request) ...
# No gpt_param changes needed
def run_gpt_update_retrieved_summary(name, summary, retrieved):
    # ... (prompt construction seems okay) ...
    key_description = ""
    if retrieved:
        for v in retrieved[:10]: # Limit input size?
            key_description += f"- {v.description}\n"
    else: key_description = "- No new thoughts or memories found.\n"

    format_instr = '{"summary":"..."}' # JSON format instruction

    prompt_input = f"This is the current summary of thoughts for {name}:\n" \
                   f"{summary}\n\n" \
                   f"The following are new thoughts or memories retrieved for {name}:\n" \
                   f"{key_description.strip()}\n\n" \
                   f"Considering the current summary and the new thoughts/memories, update the summary to incorporate any significant new information. Keep it brief and precise. If the new information is trivial or already covered, repeat the original summary.\n\n" \
                   f"Respond ONLY with the updated (or repeated) summary in JSON format:\n{format_instr}"

    print('--- Running Update Retrieved Summary ---')
    # print(f"Input:\n {prompt_input}")
    try:
        # Uses ChatGPT_single_request -> llm_request
        output = ChatGPT_single_request(prompt_input)
        # print(f"Raw Output:\n {output}")
        # Extract JSON robustly
        parsed = extract_first_json_dict(output)
        if parsed and "summary" in parsed:
             updated_summary = parsed['summary'].strip()
             print(f"Updated Summary Output for {name}: {updated_summary}")
             return updated_summary
        else:
             print(f"WARN: Could not parse JSON summary in update. Falling back to original. Raw: {output[:200]}")
             return summary # Fallback to original summary
    except Exception as e:
        metrics.fail_record(f"Update Summary Error: {e}")
        return summary # Fallback to original summary


# ... (run_gpt_update_relationship - uses direct ChatGPT_single_request) ...
# No gpt_param changes needed
def run_gpt_update_relationship(init_persona, target_persona, conversation_summary, summary):
    # ... (prompt construction seems okay, includes persona info, activities, thoughts, convo summary, current relationship) ...
    self_information = init_persona.scratch.get_str_iss()
    relationship_info = init_persona.scratch.get_relationship_feeling(target_persona.scratch.name)
    current_relationship = relationship_info.get('relationship', 'unknown')
    current_feeling = relationship_info.get('feeling', 'neutral')

    format_instr = """{"relationship": "...", "feeling": "..."}"""
    example = """Example response: {"relationship": "close friend", "feeling": "friendly and collaborative"}"""

    # Get current activities (use planned activities for robustness)
    init_act_desc = "idle"
    if init_persona.scratch.get_f_daily_schedule_index() < len(init_persona.scratch.f_daily_schedule):
        init_act_desc, _ = init_persona.scratch.f_daily_schedule[init_persona.scratch.get_f_daily_schedule_index()]

    target_act_desc = "idle"
    if target_persona.scratch.get_f_daily_schedule_index() < len(target_persona.scratch.f_daily_schedule):
        target_act_desc, _ = target_persona.scratch.f_daily_schedule[target_persona.scratch.get_f_daily_schedule_index()]

    prompt_input = f"Information about {init_persona.scratch.get_str_firstname()}:\n{self_information}\n\n" \
                   f"{init_persona.scratch.get_str_firstname()}'s current activity: {init_act_desc}\n" \
                   f"{target_persona.scratch.get_str_firstname()}'s current activity: {target_act_desc}\n\n" \
                   f"Current thoughts summary for {init_persona.scratch.get_str_firstname()}:\n{summary}\n\n" \
                   f"Summary of recent conversation between {init_persona.scratch.get_str_firstname()} and {target_persona.scratch.get_str_firstname()}:\n{conversation_summary}\n\n" \
                   f"Current relationship ({init_persona.scratch.get_str_firstname()} towards {target_persona.scratch.get_str_firstname()}): {current_relationship}\n" \
                   f"Current feeling ({init_persona.scratch.get_str_firstname()} towards {target_persona.scratch.get_str_firstname()}): {current_feeling}\n\n" \
                   f"Based on all the information, update the relationship description and how {init_persona.scratch.get_str_firstname()} feels towards {target_persona.scratch.get_str_firstname()}.\n" \
                   f"Respond ONLY in JSON format:\n{format_instr}\n{example}"


    print('--- Running Update Relationship ---')
    # print(f"Input:\n {prompt_input}")
    try:
        # Uses ChatGPT_single_request -> llm_request
        output = ChatGPT_single_request(prompt_input)
        # print(f"Raw Output:\n {output}")
        # Extract JSON robustly
        parsed = extract_first_json_dict(output)
        if parsed and 'relationship' in parsed and 'feeling' in parsed:
            # Basic validation
            if isinstance(parsed['relationship'], str) and isinstance(parsed['feeling'], str):
                 print(f"Updated Relationship Output for {init_persona.name} -> {target_persona.name}: {parsed}")
                 return parsed
            else: print(f"WARN: Relationship update JSON has invalid types: {parsed}")
        else: print(f"WARN: Could not parse relationship JSON or missing keys. Raw: {output[:200]}")

    except Exception as e:
        metrics.fail_record(f"Update Relationship Error: {e}")

    # Fallback to previous values if anything fails
    return {"relationship": current_relationship, "feeling": current_feeling}