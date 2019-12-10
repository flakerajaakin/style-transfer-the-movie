import os
import subprocess

for root, dirs, files in os.walk("/Users/thijs/projects/python/lucy"):
    for name in files:
        inputPath = os.path.join(root, name)
        outputPath = os.path.join(root, "result", "la_muse" + name)
        subprocess.call("python run_test.py --content %s --style_model models/la_muse.ckpt --output %s --max_size=1024" %(inputPath, outputPath), shell=True)
    break
