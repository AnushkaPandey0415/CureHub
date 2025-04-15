import subprocess

def main():
    print("🔧 Step 1: Training the diagnosis model...")
    subprocess.run(["python", "src/train.py"])

    print("\n🧪 Step 2: Evaluating the model on test data...")
    subprocess.run(["python", "src/evaluate.py"])

if __name__ == "__main__":
    main()