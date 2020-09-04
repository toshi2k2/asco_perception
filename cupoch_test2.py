from scipy.io import loadmat
from types import SimpleNamespace
import os, time
from os.path import exists, join, basename, splitext
import urllib.request
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import scipy.ndimage as nd
import cupoch as x3d
x3d.initialize_allocator(x3d.PoolAllocation, 1000000000)

import pyrealsense2 as rs
from read_bag import depth_filter, camera_intrinsics
from ros_detect_planes_from_depth_img.plane_detector import test_PlaneDetector_send


def plane_detection(color_image, depth_array, loop=1):
    planes_mask, planes_normal, list_plane_params = test_PlaneDetector_send(\
            img_color=color_image, img_depth=depth_array)
    for idx in range(loop-1):
        planes_mask_x, planes_normal_x, list_plane_params_x = test_PlaneDetector_send(\
            img_color=color_image, img_depth=depth_array)
        planes_mask += planes_mask_x
    planes_mask_binary = planes_mask
    r1, g1, b1 = 255, 0, 40 # Original value
    r2, g2, b2 = 1, 1, 1 # Value that we want to replace it with
    red, green, blue = planes_mask[:,:,0], planes_mask[:,:,1], planes_mask[:,:,2]
    mask = (red > r1) & (green == g1) & (blue > b1)
    planes_mask_binary[:,:,:3][mask] = [r2, g2, b2]
    planes_mask_binary = planes_mask_binary[:,:,0]
    # plane is blue (255,0,40)
    return planes_mask_binary


def run_loop():
    # Create pipeline
    pipeline = rs.pipeline()
    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    # Start streaming from file
    Pipe = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = Pipe.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create colorizer object
    colorizer = rs.colorizer()
    idx = 0
    # initial frame delay
    idx_limit = 0

    vis = x3d.visualization.Visualizer()
    vis.create_window()

    ocgd = x3d.geometry.OccupancyGrid(0.03, 512)
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    # pre_seg_mask_sum = None  # previous frame path segmentation area

    # Streaming loop
    frame_count = 0
    try:
        while True:
            idx+=1
            # Get frameset of depth
            frames = pipeline.wait_for_frames()
            # ignore first idx frames
            if idx < idx_limit:
                continue
            else:
                pass

            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            # Get color frame
            color_frame = frames.get_color_frame()
            # Get depth frame
            depth_frame = frames.get_depth_frame()
            # Get intrinsics and extrinsics

            if idx == idx_limit:
                camera_intrinsics(color_frame, depth_frame, Pipe)
            # intrinsic = x3d.camera.PinholeCameraIntrinsic(get_intrinsic_matrix(color_frame))
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            # color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            # intrinsic = x3d.camera.PinholeCameraIntrinsic((color_intrin))
            intrinsic = x3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)

            color_image = np.asanyarray(color_frame.get_data())

            depth_frame = depth_filter(depth_frame)
            depth_array = np.asarray(depth_frame.get_data())
            # Colorize depth frame to jet colormap

            # ############ Plane Detection
            # ## need to add smoothening between frames - by plane weights' variance?
            # try:
            #     ### need to add multithreading here (and maybe other methods?)
            #     planes_mask_binary = plane_detection(color_image, depth_array,\
            #         loop=3)
            # except TypeError as e:
            #     try:
            #         print("plane mask 1st error")
            #         planes_mask, planes_normal, list_plane_params = test_PlaneDetector_send(img_color=color_image, img_depth=depth_array)
            #     except TypeError as e:
            #         print("plane mask not detected-need to skip frames")
            # ##############################################
            # ## Hole filling for plane_mask (plane mask isn't binary - fixed that!)
            # planes_mask_binary = nd.binary_fill_holes(planes_mask_binary)
            # planes_mask_binary = planes_mask_binary.astype(np.uint8)
            # # Clean plane mask object detection by seg_mask
            # planes_mask_binary_3d = np.dstack((planes_mask_binary, planes_mask_binary, planes_mask_binary))
            # #############################################

            # depth_image = x3d.geometry.Image(depth_array*planes_mask_binary)
            # color_image = x3d.geometry.Image(color_image*planes_mask_binary_3d)

            depth_image = x3d.geometry.Image(depth_array)
            color_image = x3d.geometry.Image(color_image)

            rgbd_image = x3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1.0 / depth_scale,
                depth_trunc=10,
                convert_rgb_to_intensity=False)
            temp = x3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic)
            temp.transform(flip_transform)
            temp = temp.voxel_down_sample(0.03)
            ocgd.insert(temp, np.zeros(3))

            if frame_count == 0:
                vis.add_geometry(ocgd)

            vis.update_geometry(ocgd)
            vis.poll_events()
            vis.update_renderer()

            # dt1 = datetime.now()
            # process_time = dt1 - dt0
            # print("FPS: " + str(1 / process_time.total_seconds()))
            frame_count += 1

    finally:
        pipeline.stop()
        vis.destroy_window()
    return

if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file", required=True)
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    try:
        run_loop()
    finally:
        pass
    
