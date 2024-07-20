import os
import shutil
import random

# unzip kagglecatsanddogs_5340.zip into /dataset before running

def split_files(source_dir, dest_dir1, dest_dir2, split_ratio=0.8):
    # Ensure destination directories exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)

    # Get a list of all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    if not files:
        print("No files found in the source directory.")
        return

    # Shuffle the files
    random.shuffle(files)

    # Calculate split index
    split_index = int(len(files) * split_ratio)

    # Split the files into two lists
    files_for_dir1 = files[:split_index]
    files_for_dir2 = files[split_index:]

    # Copy files to the respective directories
    for file in files_for_dir1:
        src_file = os.path.join(source_dir, file)
        dest_file = os.path.join(dest_dir1, file)
        if not os.path.exists(dest_file):  # Ensure no duplicate file in dest_dir1
            shutil.copy(src_file, dest_file)

    for file in files_for_dir2:
        src_file = os.path.join(source_dir, file)
        dest_file = os.path.join(dest_dir2, file)
        if not os.path.exists(dest_file):  # Ensure no duplicate file in dest_dir2
            shutil.copy(src_file, dest_file)

    print(f"Copied {len(files_for_dir1)} files to {dest_dir1}")
    print(f"Copied {len(files_for_dir2)} files to {dest_dir2}")


# source dirs
current_dir = os.path.dirname(__file__)
cat_source_dir = current_dir+'\kagglecatsanddogs_5340\PetImages\Cat'
dog_source_dir = current_dir+'\kagglecatsanddogs_5340\PetImages\Dog'

# split cat images
split_files(cat_source_dir, current_dir+'/train\Cat', current_dir+'/validation\Cat')

# split dog images
split_files(dog_source_dir, current_dir+'/train\Dog', current_dir+'/validation\Dog')