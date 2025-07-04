import subprocess
import sys

def run_command(cmd):
    print(f"\n=== Running: {cmd} ===")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed: {cmd}")
        sys.exit(1)

def main():
    # Step 1: Prepare data
    run_command("python scripts/prepare_data.py")

    # Step 2: Run simulations and save results
    run_command("python -m scripts.main_full_run")

    # Step 3: Launch Streamlit app
    print("\nStarting Streamlit app. Open http://localhost:8501 in your browser.")
    run_command("streamlit run app/streamlit_app.py")

if __name__ == "__main__":
    main()
