import os
import sys

def startproject(project_name):
    if not project_name.isidentifier():
        print("Invalid project name. Your project name must be a valid Python identifier. Donot use hyphens or spaces. Use underscores instead.")
        sys.exit(1)
    if os.path.exists(project_name):
        print(f"Directory '{project_name}' already exists.")
        sys.exit(1)
    os.makedirs(os.path.join(project_name, "configs"))
    os.makedirs(os.path.join(project_name, "tasks"))
    os.makedirs(os.path.join(project_name, "models"))
    os.makedirs(os.path.join(project_name, "data"))
    with open(os.path.join(project_name, "main.py"), "w") as f:
        f.write("import gptmed\n\n# Your project entrypoint\n")
    print(f"Project '{project_name}' created.")
