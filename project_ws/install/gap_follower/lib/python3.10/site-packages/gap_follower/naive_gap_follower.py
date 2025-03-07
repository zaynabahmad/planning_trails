import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np

class GapFollower(Node):
    def __init__(self):
        super().__init__('gap_follower')
        
        # Parameters
        self.max_speed = 1.6
        self.max_angle_vel = 1.0
        self.t = 3.5  # threshold distance for a gap (free space)
        self.n = 30   # minimum number of consecutive points required for a valid gap
        self.robot_yaw = 0.0  # Current yaw from odometry
        
        # Subscribers & Publishers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def odom_callback(self, msg):
        """ Extracts the yaw angle from the robot's odometry """
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)

    def euler_from_quaternion(self, x, y, z, w):
        """ Converts quaternion to roll, pitch, yaw """
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return 0.0, 0.0, yaw  # roll, pitch, yaw

    def lidar_callback(self, scan_msg):
        """ Processes LiDAR data and follows the largest gap """
        ranges = list(scan_msg.ranges)
        total_rays = len(ranges)
        start_idx = 0  
        end_idx = total_rays // 2  

        filtered_ranges = ranges[start_idx:end_idx + 1]
        gap_start, gap_end = self.find_largest_gap(filtered_ranges)

        twist_msg = Twist()
        
        if gap_start is None or gap_end is None:
            self.get_logger().warn("No valid gap found! Stopping movement.")
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
        else:
            desired_heading = self.select_heading(gap_start, gap_end, scan_msg.angle_min, scan_msg.angle_increment, filtered_ranges)
            
            # Adjust heading using odometry
            corrected_heading = desired_heading - self.robot_yaw
            corrected_heading = math.atan2(math.sin(corrected_heading), math.cos(corrected_heading))

            twist_msg.linear.x = self.max_speed  
            twist_msg.angular.z = self.steering_control(corrected_heading)

        self.get_logger().info(f"cmd_vel: linear.x={twist_msg.linear.x:.2f}, angular.z={twist_msg.angular.z:.2f}")
        self.cmd_pub.publish(twist_msg)

    def find_largest_gap(self, ranges):
        """ Finds the largest free space (gap) in LiDAR scan """
        gaps = []
        start = None
        for i in range(len(ranges)):
            if ranges[i] >= self.t:
                if start is None:
                    start = i
            else:
                if start is not None:
                    gaps.append((start, i - 1))
                    start = None
        if start is not None:
            gaps.append((start, len(ranges) - 1))
        
        valid_gaps = [gap for gap in gaps if (gap[1] - gap[0] + 1) >= self.n]
        if valid_gaps:
            largest_gap = max(valid_gaps, key=lambda g: g[1] - g[0])
            return largest_gap
        elif gaps:
            largest_gap = max(gaps, key=lambda g: g[1] - g[0])
            return largest_gap
        else:
            return None, None

    def select_heading(self, gap_start, gap_end, angle_min, angle_inc, ranges):
        """ Finds the heading angle towards the farthest point in the largest gap """
        max_range_idx = max(range(gap_start, gap_end + 1), key=lambda i: ranges[i])
        best_heading = angle_min + (max_range_idx * angle_inc)

        # Convert to range [-π, π] relative to front of robot
        heading_error = best_heading - (math.pi / 2)
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        return heading_error

    def steering_control(self, heading_error, k_p=0.17):
        """ Converts heading error into angular velocity using proportional control """
        angular_vel = k_p * heading_error
        angular_vel = max(min(angular_vel, self.max_angle_vel), -self.max_angle_vel)
        return angular_vel

def main(args=None):
    rclpy.init(args=args)
    node = GapFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
