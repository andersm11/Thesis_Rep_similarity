import os

def rename_files(number: int):
    for filename in os.listdir():
        if os.path.isfile(filename):  # Ignore directories
            name, ext = os.path.splitext(filename)
            new_name = f"{name}_{number}{ext}"
            os.rename(filename, new_name)
            print(f"Renamed: {filename} → {new_name}")

# Change the number as needed
rename_files(42)  # Example: Adds "_42" before the file extension
