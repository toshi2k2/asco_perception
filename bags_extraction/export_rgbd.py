import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import pickle
from pathlib import Path

from imu_datatype import Imu

def export_rgbd_imu(bag_name):
    print("exporting bag " + bag_name)

    Path('./export/'+bag_name+'/rgb').mkdir(parents=True, exist_ok=True)
    Path('./export/'+bag_name+'/d').mkdir(parents=True, exist_ok=True)

    bag = rosbag.Bag(bag_name+'.bag')
    bridge = CvBridge()

    cnt_rgb = 0
    cnt_d = 0

    imus = []

    # for topic, msg, t in bag.read_messages(topics=['/camera/color/image_raw', '/camera/depth/image_rect_raw']):
    for topic, msg, t in bag.read_messages():
        print(topic)
        # if topic == '/camera/gyro/sample':
        #     # msg.orientation.w/x/y/z -- float
        #     # msg.orientation_covariance -- tuple
        #     # msg.angular_velocity.x/y/z -- float
        #     # msg.angular_velocity_covariance -- tuple
        #     # msg.linear_acceleration.x/y/z -- float
        #     # msg.linear_acceleration_covariance -- tuple
        #     # t.to_nsec() -- int, timestamp in nanoseconds
        #     imus.append(Imu(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation_covariance, 
        #                  msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, msg.angular_velocity_covariance, 
        #                  msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z, 
        #                  msg.linear_acceleration_covariance, t.to_nsec()))
        # elif topic == '/camera/color/image_raw':
        #     cvimg = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #     cv2.imwrite('export/'+bag_name+'/rgb/frame_'+str(cnt_rgb).zfill(4)+'_rgb_'+str(t.to_nsec())+'.png', cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR))
        #     cnt_rgb += 1
        # elif topic == '/camera/depth/image_rect_raw':
        #     cvimg = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #     cv2.imwrite('export/'+bag_name+'/d/frame_'+str(cnt_d).zfill(4)+'_d_'+str(t.to_nsec())+'.png', cvimg)
        #     cnt_d += 1

    bag.close()


    print('imu count:', len(imus))
    print('rgb image count:', cnt_rgb)
    print('depth image count:', cnt_d)

    with open('export/'+bag_name+'/imu.pkl', 'wb') as f:
        pickle.dump(imus, f)

if __name__ == '__main__':
    bags = Path('./').glob('*.bag')
    for bag in bags:
        export_rgbd_imu(str(bag)[:-4])
