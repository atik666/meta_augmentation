import os
import shutil
import glob

RootDir1 = r'/home/atik/Documents/UMAML_FSL/data/train/'
TargetFolder1 = r'/home/atik/Documents/UMAML_FSL/data/unsup_100/train/'

# dump images in a single folder

# for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
#         for name in files:
#             if name.endswith('.jpg') or name.endswith('.jpeg'):
#                 #print ("Found")
#                 SourceFolder = os.path.join(root,name)
#                 shutil.copy2(SourceFolder, TargetFolder1)
                
""""""        

# move selected number of images

for root, dirs, files in os.walk((os.path.normpath(RootDir1)), topdown=False):
        for i in dirs:
            SourceFolder = os.path.join(RootDir1,i)
            TargetFolder = os.path.join(TargetFolder1,i)
            if not os.path.exists(TargetFolder):
                os.makedirs(TargetFolder)
            for filename in glob.glob(os.path.join(SourceFolder, '*.*'))[:150]: # select number of files
                print(filename)
                shutil.copy(filename, TargetFolder)
