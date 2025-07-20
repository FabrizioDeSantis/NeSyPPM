import pm4py
import pandas as pd

def import_csv(file_path):
    event_log = pd.read_csv(file_path)
    event_log = pm4py.format_dataframe(event_log, case_id='case_id', activity_key='activity', timestamp_key='timestamp')
    return event_log

def import_xes(file_path):
    event_log = pm4py.read_xes(file_path)
    return event_log

def extract_rules_from_log(file_path):
    event_log = import_csv(file_path)
    declare_model = pm4py.discover_declare(event_log)
    return declare_model

if __name__ == "__main__":
    file_path = "data/"
    declare_model = extract_rules_from_log("path/to/your/event_log.csv")