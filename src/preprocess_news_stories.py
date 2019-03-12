import numpy as np
import os
from pathlib import Path

NEWS_ROOT = Path("./")

cnn_files = os.listdir(NEWS_ROOT/"cnn/stories")
dm_files = os.listdir(NEWS_ROOT/"dailymail/stories")

cnn_docs = []
for c in cnn_files:
    f = open(NEWS_ROOT/"cnn/stories/"+c).read()
    lines = f.replace("@headline", "")
    lines = lines.replace("\n", " ")
    cnn_docs.append(lines)
np.save(file="cnn_dump.npy", arr=np.array(cnn_docs))

dm_docs = []
for c in dm_files:
    f = open(NEWS_ROOT/"dailymail/stories/"+c).read()
    lines = f.replace("@headline", "")
    lines = lines.replace("\n", " ")
    dm_docs.append(lines)
np.save(file="dm_dump.npy", arr=np.array(dm_docs))
