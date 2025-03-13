import os

# List all files in the current directory
for filename in os.listdir():

    if filename.startswith("spatial_attention") and filename.endswith(".pth"):
        
        # Replace "spatial_attention" with "LSTM"
        new_filename = filename.replace("spatial_attention", "LSTM", 1)

        # Rename the file
        os.rename(filename, new_filename)
        print(f"Renamed: {filename} -> {new_filename}")

print("Renaming complete.")
