# Traversable surface perception

To install the dependencies, please run:

`pip3 install -r requirements.txt`

(It has been tested on Arch based Linux systems)
In case there's an error installing `pyrealsense2` library - you'd need to install it from source (or use non pip package managers)

We are using a pretrained (ADE20K dataset) model in the code, for which, run:

`pip3 install git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master`

This does not run out of the box and a modified dataset method has been written in `seg_dataset.py`. The default encoder and decoder being used are _resnet101_ and _upernet_ but can be modified in `segmentation_model_init` method in `return_pipeline`.

In case you want to render the traversable path, you'd need to install cupoch:

`pip3 install cupoch`

(Using `ctags` for understanding the workflow is highly recommended)

The main user entry file is `return_pipeline.py`:

`python3 return_pipeline.py -i <path to rosbag> -s save images (default:False) -m <mode of output> (default:0)`
There are 3 modes of output:
0: Output is in a form of video visualization
1: Output is in a form of video visualization and a list of pointclouds (output_list)
2: Only a list of pointclouds is the output

_Caution_: The list is not yet returned from the main method and the user is expected to modify the return to their own requirements. It is to be noted that having a large rosbag would result in a large pointcloud list as well. Running further processes in the while loop (especially when running real time) would be a good practice instead of using the method as a generator.

_Note 1_: The pointcloud consist only of the boundary between traversable and non-traversable areas. This can be modified to give any point cloud - the original frame or the entire traversable path. Please refer to the `return_pipleine` file and look for `rgbd_image` variable.

_Note 2_: The file is heavily commented - please refer to it for more understanding and modifications.

`pipeline2.py` file is very similar to `return_pipleine` and can be used for experimentation purposes. It only provides cv2 video visualization as an output.Please run:
`python3 pipeline2.py -i <path to rosbag> -s save images (default:False)`

In order to render the traversable path output in 3D, please install `cupoch` and use either `visual_pipe.py` or `cupoch_test.py`. The former is again similar to `pipeline2` with modifications added for 3d rendering. `cupoch_test.py` file is similat to `visual_pipe` but doesn't have implementation of the _semantic segmentaion_ model inference.
`python3 visual_pipe.py -i <path to rosbag>`

_TO-DO_: Due to lack of enough VRAM, I've been unable to fully test this file.
`
python3 cupoch_test.py -i <path to rosbag>`

`3d mat_plot.py` contains a basic example for 3d visualization of an rdb and depth image using _matplotlib_.

`read_bag.py` is an intial experimentation file which contains some methods used in our entry file. `utils.py` provides support functionality to `read_bag`.

The folder _intial_exp_labs_ contains jupyter notebooks which were used in experimentation with depth flters, plane detection, etc. - these are not commented.

The folder _bags_extraction_ contains code files for converting rosbags to rgb and depth images without using pyrealsense library. 

Both, _ros_detect_planes_from_depth_img_ and _RGBDPlaneDetection_ are folders containing code for plane detection while the former is RANSAC based, the latter is using agglomerative hierarchical clustering. Our code primarily uses the former method, which has been modified trivially to sync better with our method. You can modify and tune the respective values in the yaml files under `ros_detect.../config/`.

Please refer to the writeup under _documentation_ for further explanation of methadology.


_To-Do_: 
* Sparsify and convert the segmentation model to half-tensors. Knowedge ditilaation can also be used to significantly reduce the memory usage during inference.
* Need to parallelize and optimize most of the code in entry file.
* Add temporal filter to `depth_filter` method in `read_bag`.
* Add methodolgy report.

