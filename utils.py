import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def rgbd(color_image, depth_image):
    # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image), o3d.geometry.Image(
    # np.asanyarray(aligned_depth_frame.get_data())))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image))
    return rgbd_image

def point_cloud(rgbd_image):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, 
        o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd], zoom=0.5)
    return pcd

def plane_smooth(prev_cnt, now_cnt, now_idx, now_image):
    # print(prev_cnt)
    if prev_cnt[0]-now_cnt[0] > prev_cnt[0]/15:
        try:
            now_image[now_image==now_idx[1]]=now_idx[0]
        except IndexError as e:
            print("pretty bad depth and planar results")
            pass
    return now_image

def non_max_suppress(image, idx):
    image[image!=idx]=0
    return image

def sorted_plane_segs(image):
    idx, counts = np.unique(image, return_counts=True)
    count_sort_ind = np.argsort(-counts)
    idx = idx[count_sort_ind]
    counts = counts[count_sort_ind]
    idx, counts = list(idx), list(counts)
    try:
        x = idx.index(0)
        del idx[x]
        del counts[x]
    except ValueError as e:
        pass
    return idx, counts