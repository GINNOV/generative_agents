import re
from collections import defaultdict

def analyze_log(log_file_path):
    """
    Parses a log file to identify and summarize major issues.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        dict: A dictionary summarizing the counts of major issues found.
               Returns None if the file cannot be read.
    """
    issues = defaultdict(int)
    # Store details for context
    timeout_details = []
    wakeup_warn_details = []
    json_error_details = []
    fail_safe_details = []
    low_activity_details = []

    # Regular expressions to match specific issues
    # Making this more specific to the daily plan timeout
    timeout_pattern = re.compile(r"LLM API ERROR \(llama, Model: llama3:latest\): Timeout.*generate_first_daily_plan", re.IGNORECASE)
    fail_safe_pattern = re.compile(r"Legacy FAIL SAFE TRIGGERED", re.IGNORECASE)
    # Making this more specific to the wake_up_hour function
    wakeup_warn_pattern = re.compile(r"WARN: Could not extract hour number from wake_up response", re.IGNORECASE)
    json_error_pattern = re.compile(r"ERROR Cleaning JSON", re.IGNORECASE)
    low_activity_pattern = re.compile(r"Warning: generate_hourly_schedule -> activities num \(\d+\) is less than 7", re.IGNORECASE)

    # Patterns to identify the persona associated with an issue block
    persona_block_start_pattern = re.compile(r"=== persona/prompt_template/v\d+/.*/(wake_up_hour|daily_planning)_.*\.txt")
    persona_name_pattern = re.compile(r"~~~ persona\s*-+\s*(.*)")

    current_persona = "Unknown"
    persona_context = "Unknown" # Store last known persona for errors occurring outside blocks

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines() # Read all lines to look backwards if needed

        for line_num, line in enumerate(lines, 1):
            line_strip = line.strip()

            # Update current persona when entering a relevant block
            if persona_block_start_pattern.search(line_strip):
                # Look for the persona name within the next few lines of this block
                temp_persona = "Unknown"
                for i in range(line_num, min(line_num + 5, len(lines))):
                    match = persona_name_pattern.match(lines[i].strip())
                    if match:
                        temp_persona = match.group(1).strip()
                        break
                current_persona = temp_persona
                persona_context = current_persona # Update overall context

            # Check for issues, associating with the current_persona if inside a block,
            # or the last known persona_context otherwise.
            persona_for_error = current_persona if current_persona != "Unknown" else persona_context

            # Check for issues
            if timeout_pattern.search(line_strip):
                issues['llama_timeout_planning'] += 1
                timeout_details.append(f"Line {line_num} (Persona Context: {persona_for_error}): {line_strip}")
                current_persona = "Unknown" # Reset after error until next block
            elif fail_safe_pattern.search(line_strip):
                issues['fail_safe_triggered'] += 1
                fail_safe_details.append(f"Line {line_num} (Persona Context: {persona_for_error}): {line_strip}")
                current_persona = "Unknown"
            elif wakeup_warn_pattern.search(line_strip):
                issues['wakeup_hour_warn'] += 1
                # Try to find the persona just before this warning
                temp_persona_wake = persona_for_error # Default
                for i in range(max(0, line_num-6), line_num-1):
                     if "GNS FUNCTION: <generate_wake_up_hour>" in lines[i]:
                         # Look for persona name right after this line
                         for j in range(i+1, min(i+6, line_num)):
                             if "Prompt: Name: " in lines[j]:
                                 name_match = re.search(r"Prompt: Name: (.*?)\n", lines[j])
                                 if name_match:
                                     temp_persona_wake = name_match.group(1).strip()
                                     break
                         break # Exit outer loop once GNS found
                wakeup_warn_details.append(f"Line {line_num} (Persona Context: {temp_persona_wake}): {line_strip}")
                current_persona = "Unknown"
            elif json_error_pattern.search(line_strip):
                issues['json_cleaning_error'] += 1
                json_error_details.append(f"Line {line_num} (Persona Context: {persona_for_error}): {line_strip}")
                current_persona = "Unknown"
            elif low_activity_pattern.search(line_strip):
                issues['low_activity_warn'] += 1
                # Try to find the persona associated with this warning block end
                temp_persona_low = persona_for_error # Default
                for i in range(max(0, line_num-10), line_num-1):
                     if "=== END ===" in lines[i] and "daily_planning" in lines[i-1]:
                        # Look backwards for persona name
                        for j in range(i, max(0, i-15), -1):
                            match = persona_name_pattern.match(lines[j].strip())
                            if match:
                                temp_persona_low = match.group(1).strip()
                                break
                        break # Exit outer loop
                low_activity_details.append(f"Line {line_num} (Persona Context: {temp_persona_low}): {line_strip}")
                current_persona = "Unknown" # Reset after block end

    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # Print Summary Report
    print("-" * 30)
    print(f"Log Analysis Report for {log_file_path}")
    print("-" * 30)
    if not issues:
        print("No major issues identified.")
        return issues

    print("\nSummary Counts:")
    for issue, count in issues.items():
        print(f"- {issue.replace('_', ' ').title()}: {count}")

    # Print details
    print("\nDetails (Max 10 per category):")
    if issues['llama_timeout_planning'] > 0:
        print("\n--- Llama Planning Timeouts ---")
        for detail in timeout_details[:10]:
             print(detail)
        if len(timeout_details) > 10:
             print(f"... and {len(timeout_details) - 10} more occurrences.")

    if issues['fail_safe_triggered'] > 0:
        print("\n--- Fail-Safe Triggers ---")
        for detail in fail_safe_details[:10]:
             print(detail)
        if len(fail_safe_details) > 10:
             print(f"... and {len(fail_safe_details) - 10} more occurrences.")

    if issues['wakeup_hour_warn'] > 0:
        print("\n--- Wake-up Hour Warnings ---")
        for detail in wakeup_warn_details[:10]:
            print(detail)
        if len(wakeup_warn_details) > 10:
            print(f"... and {len(wakeup_warn_details) - 10} more occurrences.")

    if issues['json_cleaning_error'] > 0:
        print("\n--- JSON Cleaning Errors ---")
        for detail in json_error_details[:10]:
            print(detail)
        if len(json_error_details) > 10:
            print(f"... and {len(json_error_details) - 10} more occurrences.")

    if issues['low_activity_warn'] > 0:
        print("\n--- Low Activity Warnings ---")
        for detail in low_activity_details[:10]:
            print(detail)
        if len(low_activity_details) > 10:
            print(f"... and {len(low_activity_details) - 10} more occurrences.")

    print("-" * 30)

    return issues