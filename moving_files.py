import os
import shutil

RootDir1 = r'/home/atik/Documents/Ocast/borescope-adr-lm2500-data-develop/Processed/wo_Dup/train/'
TargetFolder = r'/home/atik/Documents/Ocast/borescope-adr-lm2500-data-develop/Processed/wo_Dup/unlabelled/'

for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
        for name in files:
            if name.endswith('.jpg') or name.endswith('.jpeg'):
                print ("Found")
                SourceFolder = os.path.join(root,name)
                shutil.copy2(SourceFolder, TargetFolder)