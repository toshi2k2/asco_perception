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


def run_loop(bag_path, seg_model, seg_opts, save_images=False):
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

    # Create opencv window to render image in
    cv2.namedWindow("Full Stream", cv2.WINDOW_NORMAL)
    
    # Create colorizer object
    colorizer = rs.colorizer()
    idx = 0
    # initial frame delay
    idx_limit = 30

    pre_seg_mask_sum = None  # previous frame path segmentation area

    # Streaming loop
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

            color_image = np.asanyarray(color_frame.get_data())

            ### Add Segmentation part here ###
            pred = test(color_image, seg_model, seg_opts)

            # pavement, floor, road, earth/ground, field, path, dirt/track
            seg_mask = (pred==11) | (pred==3) | (pred==6) | (pred==13) | (pred==29) | (pred==52) | (pred==91)#.astype(np.uint8)  
            
            if idx == idx_limit: # 1st frame detection needs to be robust
                pre_seg_mask_sum = np.sum(seg_mask)
            # checking for bad detection
            new_seg_sum = np.sum(seg_mask)
            diff = abs(new_seg_sum-pre_seg_mask_sum)
            # if diff > pre_seg_mask_sum/15:  # smoothening between segmentation outputs - seems like a bad idea since the model inputs are not connected between timesteps
            #     seg_mask = np.ones_like(pred).astype(np.uint8) # need to add depth (5mt) criterea for calculation for robustness
            #     del new_seg_sum
            # else:
            pre_seg_mask_sum = new_seg_sum
            ### mask Hole filling
            seg_mask = nd.binary_fill_holes(seg_mask).astype(int)
            seg_mask = seg_mask.astype(np.uint8)
                #####
            seg_mask_3d = np.dstack((seg_mask, seg_mask, seg_mask))

            pred_color = colorEncode(pred, loadmat(os.path.join(model_folder, 'color150.mat'))['colors'])
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
                    loop=5)
            except TypeError as e:
                try:
                    print("plane mask 1st error")
                    planes_mask, planes_normal, list_plane_params = test_PlaneDetector_send(img_color=color_image*seg_mask_3d, img_depth=depth_array*seg_mask)
                except TypeError as e:
                    print("plane mask not detected-skipping frame")
                    continue
                    ## removed this part
                    planes_mask = np.ones_like(depth_array).astype(np.uint8)
                    planes_mask = np.dstack((planes_mask, planes_mask, planes_mask))
            ##############################################
            ## Hole filling for plane_mask (plane mask isn't binary - fixed that!)
            planes_mask_binary = nd.binary_fill_holes(planes_mask_binary)
            planes_mask_binary = planes_mask_binary.astype(np.uint8)
            # Clean plane mask object detection by seg_mask
            planes_mask_binary *= seg_mask
            planes_mask_binary_3d = np.dstack((planes_mask_binary, planes_mask_binary, planes_mask_binary))
            edges = planes_mask_binary - nd.morphology.binary_dilation(planes_mask_binary) # edges calculation
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            #############################################

            # for cv2 output
            pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # if save_images == True:
            #     cv2.imwrite("data_/color/frame%d.png" % idx, color_image)     # save frame as JPEG file
            #     cv2.imwrite("data_/depth/frame%d.png" % idx, depth_array)     # save frame as JPEG file
            #     cv2.imwrite("data_/color_depth/frame%d.png" % idx, depth_color_image)     # save frame as JPEG file
            #     cv2.imwrite("data_/thresholded_color/frame%d.png" % idx, thresholded_color_image)     # save frame as JPEG file
            #     # cv2.imwrite("data_/thresholded_depth/frame%d.png" % idx, thresholded_depth_image)     # save frame as JPEG file

            # # Blending images
            alpha = 0.2
            beta = (1.0 - alpha)
            dst = cv2.addWeighted(color_image, alpha, pred_color, beta, 0.0)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
            # res = cv2.morphologyEx(planes_mask,cv2.MORPH_OPEN,kernel)
            dst2 = cv2.addWeighted(depth_color_image, alpha, color_image, beta, 0.0)

            ## for displaying seg_mask
            seg_mask = (np.array(seg_mask)*255).astype(np.uint8)
            seg_mask = cv2.cvtColor(seg_mask,cv2.COLOR_GRAY2BGR)
            ##################################

            ### delete later
            final_output = color_image*planes_mask_binary_3d
            mask = (final_output[:,:,0] ==0) & (final_output[:,:,1] ==0) & (final_output[:,:,2] ==0)
            final_output[:,:,:3][mask] = [255, 255, 255]
            ######

            # if np.sum(planes_mask) == depth_array.shape[0]*depth_array.shape[1]:
            #     image_set1 = np.vstack((dst, color_image))
            # else:
            image_set1 = np.vstack((color_image, depth_color_image))   
            # image_set2 = np.vstack((planes_mask_binary_3d*255, seg_mask))
            image_set2 = np.vstack((dst, final_output))
            # image_set2 = np.vstack((edges, final_output))
            combined_images = np.concatenate((image_set1, image_set2), axis=1)
            if save_images == True:
                cv2.imwrite( "./meeting_example/frame%d.png" % idx, combined_images)
            try:
                cv2.imshow('Full Stream', combined_images)
            except TypeError as e:
                print(idx, e)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    # if save_images == True:
    #     pkl.dump( threshold_mask, open( "data_/depth_threshold.pkl", "wb" ) )
    #     print("Mask pickle saved")
    return

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

    seg_mdel, options = segmentation_model_init()

    try:
        run_loop(args.input, seg_model=seg_mdel, seg_opts=options, save_images=args.save)
    finally:
        pass
    
