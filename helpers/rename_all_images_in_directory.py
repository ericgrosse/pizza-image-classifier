import os

i = 1
for filename in os.listdir("."):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        new_filename = f"pizza-{i}.{filename.split('.')[-1]}"
        os.rename(filename, new_filename)
        i += 1
