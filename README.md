# CAPTURe
Code for CAPTURe

# Setup

## Package
```bash
pip install requirements.txt
```

## Code for CAPTURe dataset
Zip for CAPTURe real occluded split is included. Image names are the same as FSC-147, so the unoccluded images can be obtained by filtering from FSC-147.

Synthetic dataset generation code is included.

# Usage
Unzip real_dataset.zip in real_dataset_zip in the main directory.
```bash
export PERSONAL_OPENAI_API_KEY=[insert key]
export HF_TOKEN=[insert token[
export HF_HOME=[insert desired home directory]
export AZURE_API_ENDPOINT=[insert api endpoint]
export AZURE_API_KEY=[insert key]
cd occluded_scripts
python -m gpt.py
```
This code will run GPT-4o through Azure API. Results will save in a new folder called "occ_results"

# Evaluation
compute_smape.py contains a function that, given a results JSON, will print the sMAPE.

# Other
All scripts to run models are in occluded_scripts or unoccluded_scripts. Script to compute sMAPE after models run is also included.
