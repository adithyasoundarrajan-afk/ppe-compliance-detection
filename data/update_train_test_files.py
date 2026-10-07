import os

def update(file_path: str, root: str):
    with open(file_path, "r") as file:
        paths = file.readlines()

    updated_paths = []
    for path in paths:
        path = path.strip()
        updated_paths.append(os.path.join(root, path) + "\n")

    new_file_name = "updated_" + os.path.basename(file_path)
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    with open(new_file_path, "w") as file:
        file.writelines(updated_paths)

    print(f"Updated file created successfully: {new_file_path}")

def main():
    # Use relative paths for the local project structure
    root_dir = "dataset/"
    train_file = "dataset/train_files.txt"
    test_file = "dataset/test_files.txt"
    
    if os.path.exists(train_file):
        update(file_path=train_file, root=root_dir)
    if os.path.exists(test_file):
        update(file_path=test_file, root=root_dir)

if __name__ == "__main__":
    main()