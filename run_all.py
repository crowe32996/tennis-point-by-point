import subprocess
import sys
import os

# Use the current Python executable
PYTHON_PATH = sys.executable

def run_command(cmd):
    print(f"\n=== Running: {cmd} ===")
    # Replace "python" with the full Python path
    if cmd.startswith("python"):
        cmd = cmd.replace("python", f'"{PYTHON_PATH}"', 1)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed: {cmd}")
        sys.exit(1)

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    print("THIS IS THE CPU COUNT:", os.cpu_count())

    requirements_path = os.path.join(PROJECT_ROOT, "requirements.txt").replace("\\", "/")
    print("\nInstalling dependencies from requirements.txt...")
    run_command(f'"{PYTHON_PATH}" -m pip install --upgrade --force-reinstall -r "{requirements_path}"')

    # Step 1: Prepare data
    prep_script = os.path.join(PROJECT_ROOT, "scripts", "prepare_data.py").replace("\\", "/")
    run_command(f'"{PYTHON_PATH}" "{prep_script}"')

    # Step 2: Run simulations
    sim_script = os.path.join(PROJECT_ROOT, "scripts", "run_simulations.py").replace("\\", "/")
    run_command(f'"{PYTHON_PATH}" "{sim_script}"')

    # Step 3: Launch Streamlit app
    streamlit_app = os.path.join(PROJECT_ROOT, "app", "streamlit_app.py").replace("\\", "/")
    print("\nStarting Streamlit app. Open http://localhost:8501 in your browser.")
    run_command(f'streamlit run "{streamlit_app}"')

if __name__ == "__main__":
    main()
