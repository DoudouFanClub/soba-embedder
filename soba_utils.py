import os

dir1 = ''
dir2 = ''

for filename in os.listdir(dir1):
    filename_no_ext = os.path.splitext(filename)[0]
    for file in os.listdir(dir2):
        if file.startswith(filename_no_ext):
            os.remove(os.path.join(dir2, file))
            print(f"Removed file {file} from directory {dir2}") #15505 31953