import os

"""reformats all the names of groundtruth labels for bbox texts"""

# Set the directory you want to change the filenames in
directory = "D:\photos\RCNN4\BBOXES"
# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file ends with .txt
    if filename.endswith('.txt'):
        # Split the filename to get the new name
        new_name = filename.split('.jpg')[0] + '.txt'
        # Create the full old and new file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_name)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} to {new_file}')

print('All filenames have been changed.')