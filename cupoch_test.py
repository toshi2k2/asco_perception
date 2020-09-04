import sys, os
import argparse
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
from ros_detect_planes_from_depth_img.plane_detector import test_PlaneDetector_send
import scipy.ndimage as nd
from read_bag import depth_filter

from datetime import datetime
import cupoch as x3d
x3d.initialize_allocator(x3d.PoolAllocation, 1000000000)


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = x3d.camera.PinholeCameraIntrinsic(1280, 720, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out

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

def main():

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    rs.config.enable_device_from_file(config, args.input)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    vis = x3d.visualization.Visualizer()
    vis.create_window()

    # ocgd = x3d.geometry.OccupancyGrid(0.05, 512)
    ocgd = x3d.geometry.OccupancyGrid()
    ocgd.visualize_free_area = False
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    prev_rgbd_image = None
    option = x3d.odometry.OdometryOption()
    cur_trans = np.identity(4)

    # Streaming loop
    frame_count = 0
    try:
        while True:

            dt0 = datetime.now()

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            aligned_depth_frame = depth_filter(aligned_depth_frame)
            color_frame = aligned_frames.get_color_frame()
            intrinsic = x3d.camera.PinholeCameraIntrinsic(
                get_intrinsic_matrix(color_frame))

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_temp = np.array(aligned_depth_frame.get_data())
            color_temp = np.asarray(color_frame.get_data())
            

            ############ Plane Detection
            ## need to add smoothening between frames - by plane weights' variance?
            try:
                ### need to add multithreading here (and maybe other methods?)
                planes_mask_binary = plane_detection(color_temp, depth_temp,\
                    loop=3)
            except TypeError as e:
                try:
                    print("plane mask 1st error")
                    planes_mask, planes_normal, list_plane_params = test_PlaneDetector_send(img_color=color_temp, img_depth=depth_temp)
                except TypeError as e:
                    print("plane mask not detected-skipping frame")
                    continue
            ##############################################
            ## Hole filling for plane_mask (plane mask isn't binary - fixed that!)
            planes_mask_binary = nd.binary_fill_holes(planes_mask_binary)
            planes_mask_binary = planes_mask_binary.astype(np.uint8)
            # Clean plane mask object detection by seg_mask
            planes_mask_binary_3d = np.dstack((planes_mask_binary, planes_mask_binary, planes_mask_binary))
            #############################################

            depth_image = x3d.geometry.Image(depth_temp*planes_mask_binary)
            color_image = x3d.geometry.Image(color_temp*planes_mask_binary_3d)


            rgbd_image = x3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image)
            if not prev_rgbd_image is None:
                res, odo_trans, _ = x3d.odometry.compute_rgbd_odometry(
                                prev_rgbd_image, rgbd_image, intrinsic,
                                np.identity(4), x3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                if res:
                    cur_trans = np.matmul(cur_trans, odo_trans)

            prev_rgbd_image = rgbd_image
            temp = x3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic)
            temp.transform(np.matmul(cur_trans, flip_transform))
            temp = temp.voxel_down_sample(0.05)
            ocgd.insert(temp, cur_trans[:3, 3])

            if frame_count == 0:
                vis.add_geometry(ocgd)

            vis.update_geometry(ocgd)
            vis.poll_events()
            vis.update_renderer()

            dt1 = datetime.now()
            process_time = dt1 - dt0
            print("FPS: " + str(1 / process_time.total_seconds()))
            frame_count += 1

    finally:
        pipeline.stop()
    vis.destroy_window()

if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file", required=True)
    parser.add_argument("-s", "--save", type=bool, help="Save video", default=False)
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
        main()
    finally:
        pass