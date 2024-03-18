import os

def create_folders(start, end, base_path):
    """
    Creates folders with names 'run_x' where x is a number from start to end, inclusive.
    
    Args:
    - start (int): The starting number for the folder names.
    - end (int): The ending number for the folder names.
    - base_path (str): The base path where the folders will be created.
    """
    for i in range(start, end + 1):
        folder_name = f"run_{i}"
        folder_path = os.path.join(base_path, folder_name)
        
        try:
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        except FileExistsError:
            print(f"Folder already exists: {folder_path}")


base_path = r'data\csv_training' 

create_folders(37, 72, base_path)