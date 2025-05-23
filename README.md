# NeSyPMM

This repository contains the code for NeSyPPM, a project exploring the integration of Neuro-Symbolic AI for Predictive Process Monitoring.

## Files

*   **`knowledge_base.py`**: Contains the logical rules used for training the neuro-symbolic LTN.
*   **`main_bpi12.py`**: Contains the code for evaluating the approach on the BPIC2012 event log.
*   **`main_bpi17.py`**: Contains the code for evaluating the approach on the BPIC2017 event log.
*   **`main_sepsis.py`**: Contains the code for evaluating the approach on the SEPSIS event log.
*   **`main_traffic.py`**: Contains the code for evaluating the approach on the TRAFFIC FINES dataset.
*   **`data/dataset.py`**: Dataset class.
*   **`metrics.py`**: Contains the code for computing the metrics for the different configurations.
*   **`plot_results.py`**: Script for plotting results with radar charts.
*   **`logs`**: Folder that contains the scripts used to preprocess the event logs.
*   **`model/lstm.py`**: Containts the architecture used for the LSTM/LTN.