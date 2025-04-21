import re
import logging
import argparse
from collections import defaultdict

def extract_persona_from_block(lines, start_index):
    persona_name_pattern = re.compile(r"~~~ persona\s*-+\s*(.*)")
    for i in range(start_index, min(start_index + 5, len(lines))):
        match = persona_name_pattern.match(lines[i].strip())
        if match:
            return match.group(1).strip()
    return "Unknown"

def analyze_log(log_file_path):
    """
    Parses a log file to identify and summarize major issues.
    
    Args:
        log_file_path (str): The path to the log file.

    Returns:
        dict: A dictionary summarizing the counts of major issues found.
               Returns None if the file cannot be read.
    """
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    issues = defaultdict(int)
    timeout_details = []
    wakeup_warn_details = []
    json_error_details = []
    fail_safe_details = []
    low_activity_details = []

    timeout_pattern = re.compile(r"LLM API ERROR \(llama, Model: llama3:latest\): Timeout.*generate_first_daily_plan", re.IGNORECASE)
    fail_safe_pattern = re.compile(r"Legacy FAIL SAFE TRIGGERED", re.IGNORECASE)
    wakeup_warn_pattern = re.compile(r"WARN: Could not extract hour number from wake_up response", re.IGNORECASE)
    json_error_pattern = re.compile(r"ERROR Cleaning JSON", re.IGNORECASE)
    low_activity_pattern = re.compile(r"Warning: generate_hourly_schedule -> activities num \(\d+\) is less than 7", re.IGNORECASE)

    persona_block_start_pattern = re.compile(r"=== persona/prompt_template/v\d+/.*/(wake_up_hour|daily_planning)_.*\\.txt")
    persona_name_pattern = re.compile(r"~~~ persona\s*-+\s*(.*)")

    current_persona = "Unknown"
    persona_context = "Unknown"

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line_strip = line.strip()

            if persona_block_start_pattern.search(line_strip):
                current_persona = extract_persona_from_block(lines, line_num)
                persona_context = current_persona

            persona_for_error = current_persona if current_persona != "Unknown" else persona_context

            if timeout_pattern.search(line_strip):
                issues['llama_timeout_planning'] += 1
                timeout_details.append(f"Line {line_num} (Persona Context: {persona_for_error}): {line_strip}")
                current_persona = "Unknown"

            elif fail_safe_pattern.search(line_strip):
                issues['fail_safe_triggered'] += 1
                fail_safe_details.append(f"Line {line_num} (Persona Context: {persona_for_error}): {line_strip}")
                current_persona = "Unknown"

            elif wakeup_warn_pattern.search(line_strip):
                issues['wakeup_hour_warn'] += 1
                temp_persona_wake = persona_for_error
                for i in range(max(0, line_num-6), line_num-1):
                    if "GNS FUNCTION: <generate_wake_up_hour>" in lines[i]:
                        for j in range(i+1, min(i+6, line_num)):
                            if "Prompt: Name: " in lines[j]:
                                name_match = re.search(r"Prompt: Name: (.*?)\\n", lines[j])
                                if name_match:
                                    temp_persona_wake = name_match.group(1).strip()
                                    break
                        break
                wakeup_warn_details.append(f"Line {line_num} (Persona Context: {temp_persona_wake}): {line_strip}")
                current_persona = "Unknown"

            elif json_error_pattern.search(line_strip):
                issues['json_cleaning_error'] += 1
                json_error_details.append(f"Line {line_num} (Persona Context: {persona_for_error}): {line_strip}")
                current_persona = "Unknown"

            elif low_activity_pattern.search(line_strip):
                issues['low_activity_warn'] += 1
                temp_persona_low = persona_for_error
                for i in range(max(0, line_num-10), line_num-1):
                    if "=== END ===" in lines[i] and "daily_planning" in lines[i-1]:
                        for j in range(i, max(0, i-15), -1):
                            match = persona_name_pattern.match(lines[j].strip())
                            if match:
                                temp_persona_low = match.group(1).strip()
                                break
                        break
                low_activity_details.append(f"Line {line_num} (Persona Context: {temp_persona_low}): {line_strip}")
                current_persona = "Unknown"

    except FileNotFoundError:
        logging.error(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
        return None

    logging.info("-" * 30)
    logging.info(f"Log Analysis Report for {log_file_path}")
    logging.info("-" * 30)

    if not issues:
        logging.info("No major issues identified.")
        return issues

    logging.info("\nSummary Counts:")
    for issue, count in issues.items():
        logging.info(f"- {issue.replace('_', ' ').title()}: {count}")

    logging.info("\nDetails (Max 10 per category):")
    def print_details(title, details):
        if details:
            logging.info(f"\n--- {title} ---")
            for detail in details[:10]:
                logging.info(detail)
            if len(details) > 10:
                logging.info(f"... and {len(details) - 10} more occurrences.")

    print_details("Llama Planning Timeouts", timeout_details)
    print_details("Fail-Safe Triggers", fail_safe_details)
    print_details("Wake-up Hour Warnings", wakeup_warn_details)
    print_details("JSON Cleaning Errors", json_error_details)
    print_details("Low Activity Warnings", low_activity_details)

    logging.info("-" * 30)
    return issues

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze log files for major issues.")
    parser.add_argument("log_file_path", help="Path to the log file")
    args = parser.parse_args()
    analyze_log(args.log_file_path)
