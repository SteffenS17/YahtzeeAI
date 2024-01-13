import os
import ast

def get_imports(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), file_path)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    
    return imports

def find_unique_libraries(folder_path):
    python_files = [f for f in os.listdir(folder_path) if f.endswith('.py')]
    libraries = set()

    for file in python_files:
        file_path = os.path.join(folder_path, file)
        imports = get_imports(file_path)
        libraries.update(imports)

    return libraries

def write_to_requirements_txt(libraries, output_file='requirements.txt'):
    with open(output_file, 'w') as file:
        for library in sorted(libraries):
            file.write(f"{library}\n")

if __name__ == "__main__":
    folder_path = os.getcwd()  # Replace with your folder path
    libraries_used = find_unique_libraries(folder_path)
    write_to_requirements_txt(libraries_used)
    print("Unique libraries have been saved to 'requirements.txt'")
