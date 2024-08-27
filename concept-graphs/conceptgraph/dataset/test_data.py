import pickle
import os
import numpy as np

test_data = "viewpoint_position.pkl"
# read the test data
with open("/home/lg1/lujia/VLN_HGT/17DRP5sb8fy_object_global_pos.pkl", 'rb') as f:
    data = pickle.load(f)
path_list_dir = "/home/lg1/lujia/VLN_HGT/17DRP5sb8fy_grid_global_pos_org.pkl"
with open(path_list_dir, 'rb') as f:
    path_list = pickle.load(f)
# data = data['viewpoints']
print(data)
print(path_list)
new_data_format = []
for data_element in data:
    data_element = data_element.cpu().numpy()
    print(data_element)
    new_data_format.append(data_element)
test_data_new = "/home/lg1/peteryu_workspace/BEV_HGT_VLN/concept-graphs/conceptgraph/dataset/17DRP5sb8fy_grid_global_pos_org.pkl"
with open(test_data_new, 'wb') as f:
    pickle.dump(new_data_format, f)