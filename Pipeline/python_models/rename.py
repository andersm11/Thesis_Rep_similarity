import os

# Define the renaming rules
rename_map = {
    "Shallow_model_temponly": "ShallowFBCSP",
    "ShallowLSTM_model_temponly": "ShallowLSTM",
    "ShallowRNN_model_temponly": "ShallowRNN"
}

# List all files in the current directory
for filename in os.listdir("."):
    if filename.endswith(".npy") and "_vs_" in filename:
        original_name = filename
        new_name = filename
        for old, new in rename_map.items():
            new_name = new_name.replace(old, new)
        if new_name != original_name:
            print(f"Renaming: {original_name} -> {new_name}")
            os.rename(original_name, new_name)
