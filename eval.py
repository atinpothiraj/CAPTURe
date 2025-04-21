import json
import numpy as np
import sys

# Function to compute MAPE
def compute_mape(filename):
    with open(filename, 'r') as f:
        json_data = json.load(f)

    total_percentage_error = 0
    count = 0
    skip = 0
    for entry in json_data:
        ground_truth = entry['ground_truth']
        if 'processed_results' in entry:
            processed_values = [entry['processed_results']]
        else:
            processed_values = [entry['results']]

        for value in processed_values:
            # Skip if predicted value is 0
            if value == -1:
                skip+=1
                total_percentage_error += 100
                count += 1
                continue

            # Compute sMAPE (Symmetric Mean Absolute Percentage Error)
            numerator = abs(value - ground_truth)
            denominator = (abs(value) + abs(ground_truth))
            smape = (numerator / denominator) * 100

        
            # Add to total percentage error
            total_percentage_error += smape
            count += 1
    
    # Calculate MAPE
    mape = total_percentage_error / count if count != 0 else 0
    return mape, skip

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compute_smape.py <path_to_results_json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    mape, skipped = compute_mape(input_path)
    print(f"sMAPE: {mape:.2f}%")
    print(f"Skipped predictions: {skipped}")

