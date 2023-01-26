import os
import shutil

RootDir1 = r'/home/atik/Documents/UMAML_FSL/data/train/'
TargetFolder = r'/home/atik/Documents/UMAML_FSL/data/unsupervised/'

for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
        for name in files:
            if name.endswith('.jpg'):
                print ("Found")
                SourceFolder = os.path.join(root,name)
                shutil.copy2(SourceFolder, TargetFolder)