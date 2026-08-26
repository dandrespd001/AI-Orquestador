import os
import subprocess

subprocess.run(["sh", "-c", "env"], check=False)
print(os.environ.get("API_KEY"))
