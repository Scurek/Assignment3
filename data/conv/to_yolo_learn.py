import os

path = "../ears/annotations/detection"
output = "data_train.txt"

web_path = "/content/TrainYourOwnYOLO/Data/Source_Images/Training_Images/"

with open(os.path.join(path, "train.txt")) as f:
    with open(os.path.join(path, output), mode="w") as o:
        # "/ content / TrainYourOwnYOLO / Data / Source_Images"
        lines = f.readlines()
        for line in lines:
            l_arr = line.split(" ")
            l_arr_name = l_arr[0].split("/")[1]
            l_arr_params = l_arr[1:5]
            o.write(web_path + "/" + l_arr_name + " " + ",".join(l_arr_params) + ",0\n")