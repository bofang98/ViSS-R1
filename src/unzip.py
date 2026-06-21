import os
import zipfile

def extract_zip_files(root_dir, target_folder):
    """
    Traverse the specified directory and all its subdirectories,
    and extract all zip files.
    Each zip file will be extracted into its containing directory.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.zip'):
                zip_path = os.path.join(dirpath, filename)
                print(f"Extracting: {zip_path}")

                # catgory
                catgory = dirpath.split('/')[-1]
                target_path = os.path.join(target_folder, catgory)
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(target_path)
                    print(f"Successfully extracted: {zip_path}")
                except Exception as e:
                    print(f"Failed to extract {zip_path}: {e}")

if __name__ == '__main__':
    # root_directory = "./src/r1-v/Video-R1-data"
    root_directory = '/root/paddlejob/workspace/env_run/gpubox03_ssd5/fangbo05/Video-R1/Video-R1-data/'
    target_folder = '/root/paddlejob/workspace/env_run/gpubox03_ssd3/fangbo05/Video-R1-data'
    extract_zip_files(root_directory, target_folder)
