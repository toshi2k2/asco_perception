from mit_semseg.config import cfg
# from mit_semseg.dataset import TestDataset
from seg_dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

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

git_repo_url = 'https://github.com/CSAILVision/semantic-segmentation-pytorch.git'
project_name = splitext(basename(git_repo_url))[0]
model_folder = 'seg_models'

def segmentation_model_init():
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    ENCODER_NAME = 'resnet101'
    DECODER_NAME = 'upernet'
    PRETRAINED_ENCODER_MODEL_URL = 'http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-%s-%s/encoder_epoch_50.pth' % (ENCODER_NAME, DECODER_NAME)
    PRETRAINED_DECODER_MODEL_URL = 'http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-%s-%s/decoder_epoch_50.pth' % (ENCODER_NAME, DECODER_NAME)

    pretrained_encoder_file = ENCODER_NAME+basename(PRETRAINED_ENCODER_MODEL_URL)
    pretrained_decoder_file = DECODER_NAME+basename(PRETRAINED_DECODER_MODEL_URL)
    encoder_path = os.path.join(model_folder, pretrained_encoder_file)
    decoder_path = os.path.join(model_folder, pretrained_decoder_file)

    if not os.path.exists(encoder_path):
        urllib.request.urlretrieve(PRETRAINED_ENCODER_MODEL_URL, encoder_path)
    if not os.path.exists(decoder_path):
        urllib.request.urlretrieve(PRETRAINED_DECODER_MODEL_URL, decoder_path)

    # options
    options = SimpleNamespace(fc_dim=2048,
                            num_class=150,
                            imgSizes = [300, 400, 500, 600],
                            imgMaxSize=1000,
                            padding_constant=8,
                            segm_downsampling_rate=8)

    # create model
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=ENCODER_NAME, weights=encoder_path,
                                        fc_dim=options.fc_dim)
    net_decoder = builder.build_decoder(arch=DECODER_NAME, weights=decoder_path,
                                        fc_dim=options.fc_dim, num_class=options.num_class, use_softmax=True)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, torch.nn.NLLLoss(ignore_index=-1))
    segmentation_module = segmentation_module.eval()
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
      segmentation_module = segmentation_module.cuda()
    return segmentation_module, options

# test on a given image
def test(test_image_name, segmentation_module, options):
  dataset_test = TestDataset([test_image_name], options, max_sample=-1)  # passing image directly to Dataset object
  
  batch_data = dataset_test[0]
  segSize = (batch_data['img_ori'].shape[0], batch_data['img_ori'].shape[1])
  img_resized_list = batch_data['img_data']
  
  scores = torch.zeros(1, options.num_class, segSize[0], segSize[1])
  if torch.cuda.is_available():
    scores = scores.cuda()

  for img in img_resized_list:
    feed_dict = batch_data.copy()
    feed_dict['img_data'] = img
    del feed_dict['img_ori']
    del feed_dict['info']
    if torch.cuda.is_available():
      feed_dict = {k: o.cuda() for k, o in feed_dict.items()}

    # forward pass
    pred_tmp = segmentation_module(feed_dict, segSize=segSize)
    scores = scores + pred_tmp / len(options.imgSizes)

    _, pred = torch.max(scores, dim=1)
    return pred.squeeze(0).cpu().numpy()

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


def run_loop(bag_path, seg_model, seg_opts):
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
    idx_limit = 30

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

            ### Add Segmentation part here ###
            pred = test(color_image, seg_model, seg_opts)

            # pavement, floor, road, earth/ground, field, path, dirt/track
            seg_mask = (pred==11) | (pred==3) | (pred==6) | (pred==13) | (pred==29) | (pred==52) | (pred==91)#.astype(np.uint8) 
            # seg_mask = np.ones((color_image.shape[0], color_image.shape[1])) 
            
            # if idx == idx_limit: # 1st frame detection needs to be robust
            #     pre_seg_mask_sum = np.sum(seg_mask)
            # checking for bad detection
            # new_seg_sum = np.sum(seg_mask)
            # diff = abs(new_seg_sum-pre_seg_mask_sum)
            # if diff > pre_seg_mask_sum/15:  # smoothening between segmentation outputs - seems like a bad idea since the model inputs are not connected between timesteps
            #     seg_mask = np.ones_like(pred).astype(np.uint8) # need to add depth (5mt) criterea for calculation for robustness
            #     del new_seg_sum
            # else:
            # pre_seg_mask_sum = new_seg_sum

            ### mask Hole filling
            seg_mask = nd.binary_fill_holes(seg_mask).astype(int)
            seg_mask = seg_mask.astype(np.uint8)
                #####
            seg_mask_3d = np.dstack((seg_mask, seg_mask, seg_mask))

            # pred_color = colorEncode(pred, loadmat(os.path.join(model_folder, 'color150.mat'))['colors'])
            ##################################

            depth_frame = depth_filter(depth_frame)
            depth_array = np.asarray(depth_frame.get_data())
            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)
            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            ############ Plane Detection
            ## need to add smoothening between frames - by plane weights' variance?
            try:
                ### need to add multithreading here (and maybe other methods?)
                planes_mask_binary = plane_detection(color_image*seg_mask_3d, depth_array*seg_mask,\
                    loop=3)
            except TypeError as e:
                try:
                    print("plane mask 1st error")
                    planes_mask, planes_normal, list_plane_params = test_PlaneDetector_send(img_color=color_image*seg_mask_3d, img_depth=depth_array*seg_mask)
                except TypeError as e:
                    print("plane mask not detected-need to skip frames")
            ##############################################
            ## Hole filling for plane_mask (plane mask isn't binary - fixed that!)
            planes_mask_binary = nd.binary_fill_holes(planes_mask_binary)
            planes_mask_binary = planes_mask_binary.astype(np.uint8)
            # Clean plane mask object detection by seg_mask
            planes_mask_binary *= seg_mask
            planes_mask_binary_3d = np.dstack((planes_mask_binary, planes_mask_binary, planes_mask_binary))
            #############################################

            depth_image = x3d.geometry.Image(depth_array*planes_mask_binary)
            color_image = x3d.geometry.Image(color_image*planes_mask_binary_3d)

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

    seg_mdel, options = segmentation_model_init()

    try:
        run_loop(args.input, seg_model=seg_mdel, seg_opts=options)
    finally:
        pass
    
