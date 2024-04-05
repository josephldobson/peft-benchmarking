import os
import pickle

results_dir = 'src/results'
output_file = 'src/pickle_contents.txt'

with open(output_file, 'w') as output:

    if os.path.isdir(results_dir):
        output.write(f"The directory '{results_dir}' was found. Looking for pickle files...\n")
        files = os.listdir(results_dir)
        
        pickle_files = [file for file in files if file.endswith('.pickle')]
        
        if not pickle_files:
            output.write("No pickle files found in the directory.\n")
        else:
            for pickle_file in pickle_files:
                file_path = os.path.join(results_dir, pickle_file)
                
                try:
                    with open(file_path, 'rb') as handle:
                        data = pickle.load(handle)
                    output.write(f"Contents of {pickle_file}:\n{data}\n\n")
                except Exception as e:
                    output.write(f"Failed to read {pickle_file}: {e}\n")
    else:
        output.write(f"The directory '{results_dir}' does not exist or is not in the current directory.\n")

print(f"The contents of the pickle files have been written to {output_file}")
