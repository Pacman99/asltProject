import os
import shutil
import numpy as np

sourceN = "/home/aslt/signData/top30ClassesCopy"
destN = "/home/aslt/signData/top30ClassesTest"
filesN = os.listdir(sourceN)

for f in filesN:
   if np.random.rand(1) < 0.2:
      shutil.move(sourceN + '\\'+ f, destN + '\\'+ f)


print(len(os.listdir(sourceN)))
print(len(os.listdir(destN)))
