import subprocess

# Specify the Python script you want to run
script_to_run = ['python', 'tools/train_model.py', 'configs/ntu120_xset/j.py',
                 '--validate', '--test-last', '--test-best']

# Number of times to run the script
num_runs = 55

for _ in range(num_runs):
    subprocess.run(script_to_run)
