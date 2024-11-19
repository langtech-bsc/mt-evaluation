import os
import json
import pandas as pd

# Path to the directory containing the subfolders
base_dir = './results'

# Initialize an empty list to hold the data
data = []

# Loop through each subfolder in the base directory
for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    
    # Check if the path is a directory
    if os.path.isdir(subfolder_path):
        # Look for JSON files in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(subfolder_path, filename)
                
                # Open and read the JSON file
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    
                    # Extract relevant information
                    for key, value in json_data['results'].items():
                        task = value.get('alias', '')
                        result_row = {
                            'model_name': subfolder,
                            'task': task,
                            'source': task.split('_')[0],
                            'target': task.split('_')[1],
                            'dataset': task.split('_', 2)[-1],
                            'bleu': value.get('bleu,none', ''),
                            'ter': value.get('ter,none', ''),
                            'chrf': value.get('chrf,none', ''),
                            'comet': value.get('comet,none', ''),
                            'comet_kiwi': value.get('comet_kiwi,none', ''),
                            'bleurt': value.get('bleurt,none', ''),
                            'xcomet': value.get('xcomet,none', ''),
                            'metricx': value.get('metricx,none', ''),
                            'metricx_qe': value.get('metricx_qe,none', '')
                        }


                        result_row.update({'file_name': filename})
                        data.append(result_row)

df = pd.DataFrame(data).drop_duplicates(['model_name', 'source', 'target', 'dataset']).round(3)

output_csv = './results_summary/results_summary.csv'
df.to_csv(output_csv, index=False)
print(f"CSV file saved as {output_csv}")