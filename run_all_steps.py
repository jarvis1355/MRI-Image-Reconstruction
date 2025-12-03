import os

scripts = [
    "step1.py",
    "step2.py",
    "step3.py",
    "step4.py",
    "step5.py",
    "step6.py",
    "step7.py"
]

for script in scripts:
    print(f"Running {script}...")
    os.system(f"python {script}")
