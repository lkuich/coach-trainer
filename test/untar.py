import os
import tarfile

data_root = "/home/lkuich/ScaledTrainer/test"

files = os.listdir(data_root)
for name in files:
    filename = name.split('.tgz')[0]
    print(filename)
    if filename:
        _tr = tarfile.open(data_root + '/' + name)
        _tr.extractall()
        break

fffiles = os.listdir(data_root + "/flower_photos")
for ffname in fffiles:
    print(ffname)
