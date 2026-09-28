
EV BATTERY PROJECT
DATA PROCESSING HANDOVER
========================

Prepared by: Darshan D

Project:
Continual Learning for Adaptive Health and Reuse Assessment
of Heterogeneous Second-Life EV Batteries


DATASET
-------

Six battery groups are represented:

g1
v4
v5
w8
w9
w10


MAIN FILE FOR ML/DL
-------------------

ml_training_dataset.csv

This is the recommended starting dataset for Tarun.

It contains cycle-level battery features together with a known
SOH value from the most recent RPT available on or before that
cycle.

Rows before the first available RPT are excluded from this
training file because their SOH is unknown.


COMPLETE ML DATASET
-------------------

final_ml_dataset.csv

Contains all cycle-level observations.

Some early observations have missing SOH because they occurred
before the first RPT measurement. These values were deliberately
NOT fabricated or imputed.


HEALTH DATASET
--------------

final_battery_health_dataset.csv

Contains:

Battery_ID
RPT
RPT_Number
Capacity_Ah
Test_Date
Reference_Capacity_Ah
SOH_Percent


SOH DEFINITION
--------------

SOH_Percent is calculated as:

SOH = Capacity_Ah / Reference_Capacity_Ah * 100

Reference_Capacity_Ah is the FIRST OBSERVED RPT capacity for
each battery in this dataset.

This is a relative/reference-based SOH indicator.

It should NOT be described as SOH relative to the manufacturer's
nominal rated capacity unless that independent rated capacity is
introduced separately.


CYCLING FEATURES
----------------

final_battery_cycling_features.csv contains battery cycling
features derived from:

Voltage
Current
Temperature
Charge capacity
Discharge capacity
Charge energy
Discharge energy
Power
Capacity ratio
Energy efficiency
Cycle count


TEMPORAL ALIGNMENT
------------------

For final_ml_dataset.csv, each cycling observation is associated
with the most recent RPT measurement that occurred on or before
the cycling observation.

This avoids assigning future health information to earlier
cycling observations.


MISSING DATA
------------

Some temperature observations are missing in the original data.

They are not silently fabricated in the final dataset.

ML preprocessing can handle/impute these values according to
the modelling method selected by Tarun.


FILES
-----

final_ml_dataset.csv
    Complete cycle-level dataset with temporal health alignment.

ml_training_dataset.csv
    SOH-labelled subset recommended for supervised ML/DL.

final_battery_health_dataset.csv
    RPT capacity and SOH progression.

final_battery_cycling_features.csv
    Battery/cycling feature dataset.

cleaned_battery_data.csv
    Cleaned point-level cycling data.

rpt_capacity_data.csv
    Extracted RPT capacity measurements.

DATA_DICTIONARY.csv
    Column definitions.


PROCESSING FLOW
---------------

Raw Excel files
       |
       v
Combined cycling data
       |
       v
Cleaned cycling data
       |
       +------------------+
       |                  |
       v                  v
Cycling features      RPT capacity
       |                  |
       |                  v
       |              Relative SOH
       |                  |
       +--------+---------+
                |
                v
       Temporal alignment
                |
                v
       final_ml_dataset.csv
                |
                v
       ml_training_dataset.csv
                |
                v
          ML / DL stage


HANDOVER
--------

Tarun should start with:

ml_training_dataset.csv

and use final_ml_dataset.csv when he needs the complete
time-series data, including observations before the first
known health measurement.
