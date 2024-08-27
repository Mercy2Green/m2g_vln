import numpy as np
import open3d as o3d
# 示例数据：随机生成
# read the .npy file
grid_points = np.load('/home/lg1/lujia/VLN_HGT/grid_global_save.npy')  # 网格点数据
objects_center = np.load('/home/lg1/lujia/VLN_HGT/object_global_save.npy')  # 包围盒中心数据
places_center = np.load('/home/lg1/lujia/VLN_HGT/places_global_save.npy')  # 地点中心数据
path_vps = np.load('/home/lg1/lujia/VLN_HGT/path_global_save.npy')

grid_points = grid_points.reshape(-1, 3) + np.array([0, 5, 0])
places_center = places_center[:, :3] + np.array([0, 10, 0])
path_vps = path_vps + np.array([0, 10, 0])
print(grid_points.shape)
print(objects_center.shape)
print(places_center.shape)
if grid_points.ndim != 2 or grid_points.shape[1] != 3:
    raise ValueError("grid_points 应该是形状为 (N, 3) 的 numpy 数组")
if grid_points.dtype != np.float64:
    grid_points = grid_points.astype(np.float64)
# 创建点云
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(grid_points)

# 可视化点云
point_cloud.paint_uniform_color([1, 0, 0])  # 红色

# 创建包围盒并添加到视图
bbox_size = 0.3  # 增大包围盒的大小
obj_bboxes = []
for center in objects_center:
    bbox = o3d.geometry.OrientedBoundingBox(center, np.eye(3), [bbox_size]*3)
    bbox.color = np.array([0, 0.5, 0])  # 深绿色
    obj_bboxes.append(bbox)
# for place in places_center:
bbox_size = 0.3  # 增大包围盒的大小
places_bboxes = []
for center in places_center:
    bbox = o3d.geometry.OrientedBoundingBox(center, np.eye(3), [bbox_size]*3)
    bbox.color = np.array([0, 0, 0.5])  # 深绿色
    places_bboxes.append(bbox)

# 创建路径包围盒并添加到视图
path_bboxes = []
coordinate_frames = []
bbox_size = 0.3  # 增大包围盒的大小

for center in path_vps:
    center = center[:3]  # 确保 center 是形状为 (3,) 的数组
    bbox = o3d.geometry.OrientedBoundingBox(center, np.eye(3), [bbox_size]*3)
    bbox.color = np.array([0, 0.5, 0.5])  # 青色
    path_bboxes.append(bbox)
    
    # 创建坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=center)
    coordinate_frames.append(coordinate_frame)

# 创建平面并添加到视图
def create_plane(center, size, color):
    plane = o3d.geometry.TriangleMesh.create_box(width=size[0], height=0.01, depth=size[1])
    plane.paint_uniform_color(color)
    plane.translate(center - np.array([size[0]/2, 0, size[1]/2]))
    return plane

planes = []
# 为每个组创建一个平面
if len(objects_center) > 0:
    plane = create_plane(objects_center[0][:3], [10, 10], [0, 0.5, 0])  # 绿色平面
    planes.append(plane)

if len(places_center) > 0:
    plane = create_plane(places_center[0][:3], [10, 10], [0, 0, 0.5])  # 蓝色平面
    planes.append(plane)
if len(grid_points) > 0:
    plane = create_plane(grid_points[0][:3], [10, 10], [0, 0.5, 0])  # 绿色平面
    planes.append(plane)

if len(path_vps) > 0:
    plane = create_plane(path_vps[0][:3], [2, 2], [0, 0.5, 0.5])  # 青色平面
    planes.append(plane)
# 初始化视窗
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加点云和包围盒到视窗
vis.add_geometry(point_cloud)
for bbox in obj_bboxes:
    vis.add_geometry(bbox)

for bbox in places_bboxes:
    vis.add_geometry(bbox)

# 添加坐标系到视窗
for frame in coordinate_frames:
    vis.add_geometry(frame)
# 添加平面到视窗
for plane in planes:
    vis.add_geometry(plane)
# # 定义按下 'q' 键时的回调函数
# def quit_callback(vis):
#     vis.destroy_window()

# # 注册 'q' 键的回调函数
# vis.register_key_callback(ord('Q'), quit_callback)

# 运行可视化窗口
vis.run()
