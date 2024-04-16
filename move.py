import os
import zipfile

original_path = "/media/m2g/Data/Datasets/dataset/v1/scans"

os.chdir(original_path)

data_path = os.listdir()

goal_path = '/media/m2g/Data/Datasets/dataset/v1/unzipped'

count_flag = 1

for path in data_path:

    print("Percent", count_flag/len(data_path))
    count_flag += 1

    os.chdir(path)

    #new_goal_path = goal_path + '/' + path

    files = os.listdir()

    for file in files:
        #unzip file to goal_path
        zip_ref = zipfile.ZipFile(file, 'r')
        zip_ref.extractall(goal_path)
        zip_ref.close()
        print("Unzipped", file)
    
    os.chdir(original_path)