# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
import open3d as o3d
from subprocess import Popen
from utils import rgbd, point_cloud, plane_smooth, non_max_suppress, sorted_plane_segs
import pickle as pkl
import matplotlib.pyplot as plt

def depth_filter(depth_frame):
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    hole_filling = rs.hole_filling_filter(2)
    spatial = rs.spatial_filter()
    
    depth_frame = depth_to_disparity.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    # frame = temporal.process(frame)
    depth_frame = disparity_to_depth.process(depth_frame)
    depth_frame = hole_filling.process(depth_frame)

    return depth_frame

def camera_intrinsics(color_frame, depth_frame, pipe_profile):
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
    depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
    color_to_depth_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)

    print("\n Depth intrinsics: " + str(depth_intrin))
    print("\n Color intrinsics: " + str(color_intrin))
    print("\n Depth to color extrinsics: " + str(depth_to_color_extrin))

    # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
    depth_sensor = pipe_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("\n\t depth_scale: " + str(depth_scale))
    return

def thresholding(depth_image, display_image, depth_scale, clip_distance_in_meters=4.5):
    grey_color = 153
    # ones_image = np.ones((display_image.shape[0], display_image.shape[1]))
    ones_image = np.ones_like(display_image)
    clipping_distance = clip_distance_in_meters / depth_scale
    depth_image = np.dstack((depth_image, depth_image, depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image > clipping_distance) | (depth_image <= 0), grey_color, display_image)
    # mask = (bg_removed != grey_color).all(axis=-1)
    mask = np.where((depth_image > clipping_distance) | (depth_image <= 0), 0, ones_image)
    # plt.imshow(mask)
    return bg_removed, mask

def read_bag(bag_path, save_images=False, plane_available=False, save_output=False, point_cloud=False):
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)
    # Configure the pipeline to stream the depth stream
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    # Start streaming from file
    Pipe = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = Pipe.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # Create opencv window to render image in
    # cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Full Stream", cv2.WINDOW_NORMAL)
    
    # Create colorizer object
    colorizer = rs.colorizer()
    idx = 0
    idx_limit = 90

    threshold_mask = []
    prev_planar_image = None

    # Streaming loop
    while True:
        idx+=1
        # Get frameset of depth
        frames = pipeline.wait_for_frames()
        # Intrinsics & Extrinsics

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
        if idx == idx_limit+1:
            camera_intrinsics(color_frame, depth_frame, Pipe)

        depth_frame = depth_filter(depth_frame)

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_array = np.asarray(depth_frame.get_data())

        # if point_cloud == False:
        #     rgbd_image = rgbd(color_image, depth_array)
        #     # point_cloud(rg_d)
        #     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, 
        #     o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        #     # Flip it, otherwise the pointcloud will be upside down
        #     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        #     o3d.visualization.draw_geometries([pcd], zoom=0.5)
        #     break

        # for cv2 output
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Thresholding image
        thresholded_color_image, mask = thresholding(np.asarray(depth_frame.get_data()), color_image, depth_scale, 4)
        thresholded_depth_image, _ = thresholding(np.asarray(depth_frame.get_data()), depth_color_image, depth_scale, 4)
        threshold_mask.append(mask)

        if save_images == True:
            cv2.imwrite("data_/color/frame%d.png" % idx, color_image)     # save frame as JPEG file
            cv2.imwrite("data_/depth/frame%d.png" % idx, depth_array)     # save frame as JPEG file
            cv2.imwrite("data_/color_depth/frame%d.png" % idx, depth_color_image)     # save frame as JPEG file
            cv2.imwrite("data_/thresholded_color/frame%d.png" % idx, thresholded_color_image)     # save frame as JPEG file
            # cv2.imwrite("data_/thresholded_depth/frame%d.png" % idx, thresholded_depth_image)     # save frame as JPEG file

        # Blending images
        alpha = 0.4
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(color_image, alpha, depth_color_image, beta, 0.0)

        # # Get plane image from c++ code
        # plane_image = Popen(['/usr/bin/foo', 'arg1', 'arg2'])
        if plane_available == True:
            plane_image = cv2.imread('./bags/subh_1/frame%d/f-plane.png' % idx, cv2.IMREAD_GRAYSCALE)
            # plane_image = cv2.imread('./bags/calibrate_inside_movingbase/frame%d/f-plane.png' % idx, cv2.IMREAD_GRAYSCALE)
            if plane_image is None:
                plane_image = cv2.imread('./bags/subh_1/frame%d/fr-plane.png' % idx, cv2.IMREAD_GRAYSCALE)
                # plane_image = cv2.imread('./bags/calibrate_inside_movingbase/frame%d/fr-plane.png' % idx, cv2.IMREAD_GRAYSCALE)

            # thresholded_plane, _ = thresholding(np.asarray(depth_frame.get_data()), plane_image, depth_scale, 4)
            # try:
            #     plane_image = cv2.cvtColor(plane_image, cv2.COLOR_BGR2GRAY)
            # except Exception as e:
            #     print(idx, e)
            #     continue
            try:
                thresholded_plane = plane_image * mask[:,:,0]
            except TypeError as e:
                print(idx, e)
                continue
            plane_idx, pix_count = sorted_plane_segs(thresholded_plane)
            # print(idx, plane_idx, pix_count)
            if idx > idx_limit:
                prev_plane_idx, prev_pix_count = sorted_plane_segs(prev_planar_image)
                # print("prev: ", idx, prev_plane_idx, prev_pix_count)
                thresholded_plane = plane_smooth(prev_pix_count, pix_count, plane_idx, thresholded_plane)
            
            thresholded_plane = non_max_suppress(thresholded_plane, plane_idx[0])
    
            if idx>idx_limit:
                prev_planar_image = thresholded_plane
            
            thresholded_plane[thresholded_plane!=0]=1
            planar_mask = np.dstack((thresholded_plane, thresholded_plane, thresholded_plane))

            # Apply opening morph
            # kernel = np.ones((5,5),np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10))
            planar_mask = cv2.morphologyEx(planar_mask, cv2.MORPH_CLOSE, kernel)

            planar_masked_color = color_image * planar_mask

            thresholded_plane[thresholded_plane!=0]=255

            try:
                thresholded_plane = cv2.cvtColor(thresholded_plane,cv2.COLOR_GRAY2BGR)
            except:
                print(thresholded_plane.shape, np.unique(thresholded_plane, return_counts=True))

        # mask = mask * 1.0
        # mask = np.dstack((mask, mask, mask)) #depth image is 1 channel, color is 3 channels
        try:
            image_set1 = np.vstack((color_image, depth_color_image))
            # image_set1 = np.vstack((color_image, dst))
            image_set2 = np.vstack((thresholded_color_image, thresholded_depth_image))
            if plane_available:
                image_set3 = np.vstack((thresholded_plane, planar_masked_color))
        except ValueError as e:
            print(color_image.shape, depth_color_image.shape)

        # Render image in opencv window
        # cv2.imshow("Depth Stream", depth_color_image)
        # cv2.imshow("Color Stream", color_image)
        # combined_images = np.concatenate((color_image, dst, depth_color_image), axis=1)
        if plane_available:
            combined_images = np.concatenate((image_set1, image_set2, image_set3), axis=1)
        else:
            combined_images = np.concatenate((image_set1, image_set2), axis=1)
        if save_output == True:
            cv2.imwrite( "./bags/subh_1_meeting/frame%d.png" % idx, combined_images)     
        try:
            cv2.imshow('Full Stream', combined_images)
        except TypeError as e:
            print(idx, e)
        # cv2.imshow("Full Stream", dst)
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
    
    if save_images == True:
        pkl.dump( threshold_mask, open( "data_/depth_threshold.pkl", "wb" ) )
        print("Mask pickle saved")
    return

if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
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
        read_bag(args.input, save_images=args.save)
    finally:
        pass