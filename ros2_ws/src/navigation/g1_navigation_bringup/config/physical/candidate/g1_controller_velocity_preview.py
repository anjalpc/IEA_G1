"""Controller-only velocity averaging. No TF or locomotion publishers."""
import argparse
import copy
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from g1_controller_velocity_window import PoseVelocityWindow

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-topic',default='/g1/state/odometry')
    parser.add_argument('--output-topic',default='/g1/navigation/controller/odometry')
    parser.add_argument('--window-s',type=float,default=.6)
    args,ros_args=parser.parse_known_args()
    if args.input_topic==args.output_topic:
        raise ValueError('Input and output topics must differ')
    rclpy.init(args=ros_args)
    node=Node('g1_controller_velocity_preview')
    duration=node.declare_parameter('filter_duration_seconds',args.window_s,ParameterDescriptor(read_only=True)).value
    source=node.declare_parameter('velocity_source','canonical_pose_window',ParameterDescriptor(read_only=True)).value
    if source!='canonical_pose_window':raise ValueError('Unsupported velocity source')
    window=PoseVelocityWindow(duration)
    publisher=node.create_publisher(Odometry,args.output_topic,5)
    def callback(message):
        if message.header.frame_id!='odom' or message.child_frame_id!='base_footprint':
            window.reset()
            return
        stamp=message.header.stamp.sec*10**9+message.header.stamp.nanosec
        pose=message.pose.pose
        q=pose.orientation
        norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w
        if not math.isfinite(norm) or not .98<=norm<=1.02:
            window.reset();return
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        try:velocity=window.add(stamp,(pose.position.x,pose.position.y),yaw)
        except ValueError:return
        if velocity is None:return
        output=copy.deepcopy(message)
        output.twist.twist.linear.x,output.twist.twist.linear.y,output.twist.twist.angular.z=velocity
        publisher.publish(output)
    subscription=node.create_subscription(Odometry,args.input_topic,callback,qos_profile_sensor_data)
    try:rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':main()
