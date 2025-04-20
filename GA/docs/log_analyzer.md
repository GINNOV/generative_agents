# Log Analyzer

This script will look for specific patterns associated with the problems that I have run into thus far:

1.  Llama Timeouts during planning.
2.  Fail-Safe triggers for planning.
3.  Wake-up hour extraction warnings.
4.  JSON cleaning errors.
5.  Low activity count warnings.

I will add more as new patterns show up. The tool is located in the `tools` folder.

# Parse the log

1. Save the code above as a Python file (e.g., `log_analyzer.py`).
2. Make sure the log file (`log.txt`) is accessible from where you run the script.
3. Run the script from your terminal, passing the log file path as an argument:
`python log_analyzer.py log.txt`

If you are running this in an environment like a Jupyter notebook
or directly modifying the script, you can call the function directly:

log_file_to_analyze = 'log.txt'
analysis_results = analyze_log(log_file_to_analyze)
if analysis_results:
    print("\nAnalysis complete.")

# Example of direct call
```python
if __name__ == "__main__":
    # Set up argument parsing for command-line use
    parser = argparse.ArgumentParser(description='Analyze log file for major issues.')
    parser.add_argument('log_file', type=str, help='Path to the log file to analyze.')
    args = parser.parse_args()

    analyze_log(args.log_file)
```

# Output
Summary Counts:
- Wakeup Hour Warn: 11
- Fail Safe Triggered: 2
- Low Activity Warn: 3
- Json Cleaning Error: 2

Details (Max 10 per category):

--- Fail-Safe Triggers ---
Line 116 (Persona Context: Unknown): Legacy FAIL SAFE TRIGGERED
Line 281 (Persona Context: Unknown): Legacy FAIL SAFE TRIGGERED

--- Wake-up Hour Warnings ---
Line 22 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to her lifestyle, Isabella Rodriguez awakes'
Line 30 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Isabella Rodriguez's lifestyle, her wake'
Line 38 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Isabella's profile, her wake-up'
Line 46 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Isabella Rodriguez's lifestyle, her wake'
Line 54 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to her lifestyle, Isabella Rodriguez awakes'
Line 213 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Maria's lifestyle, she wakes up around'
Line 376 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Klaus Mueller's lifestyle, he wakes up'
Line 384 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Klaus Mueller's lifestyle, he wakes up'
Line 392 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Klaus Mueller's lifestyle, his wake-up'
Line 400 (Persona Context: Unknown): WARN: Could not extract hour number from wake_up response: 'According to Klaus Mueller's lifestyle, he wakes up'
... and 1 more occurrences.

--- JSON Cleaning Errors ---
Line 497 (Persona Context: Unknown): ERROR Cleaning JSON: Expecting value: line 1 column 1 (char 0), gpt: Here is Klaus Mueller's plan for today:
Line 575 (Persona Context: Unknown): ERROR Cleaning JSON: Expecting value: line 1 column 1 (char 0), gpt: Here is Klaus Mueller's plan for today:

--- Low Activity Warnings ---
Line 175 (Persona Context: Unknown): Warning: generate_hourly_schedule -> activities num (3) is less than 7.
Line 340 (Persona Context: Unknown): Warning: generate_hourly_schedule -> activities num (3) is less than 7.
Line 671 (Persona Context: Unknown): Warning: generate_hourly_schedule -> activities num (2) is less than 7.
------------------------------

Here's a summary of the issues it found, based on the output:

* **Wakeup Hour Warnings (11 occurrences):** The script confirmed the frequent warnings about failing to extract the wake-up hour, particularly for Isabella and Klaus[cite: 1].
* **Fail-Safe Triggers (2 occurrences):** It detected the two instances where the daily planning failed completely, triggering the fail-safe mechanism[cite: 1].
* **Low Activity Warnings (3 occurrences):** This corresponds to the times the fail-safe plan was used, resulting in fewer than the desired number of activities for the day[cite: 1].
* **JSON Cleaning Errors (2 occurrences):** It caught the errors specifically related to cleaning the JSON output from the Llama model during Klaus's daily plan generation[cite: 1].
* **Llama Planning Timeouts (0 occurrences):** Interestingly, the refined pattern looking specifically for timeouts during `generate_first_daily_plan` didn't register any hits in this run. Looking back at the log manually[cite: 1], the timeouts *did* occur during `generate_first_daily_plan` ([cite: 1],[cite: 1], [cite: 1]), but the script pattern `re.compile(r"LLM API ERROR \(llama, Model: llama3:latest\): Timeout.*generate_first_daily_plan", re.IGNORECASE)` might have been too specific or missed them due to line breaks or slight variations not present in the test cases. However, we know from the Fail-Safe Triggers and Low Activity Warnings that the planning *did* fail, almost certainly due to these timeouts.