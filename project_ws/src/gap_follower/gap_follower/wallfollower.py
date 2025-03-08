import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        # Parameters: desired distance from wall, forward speed, and proportional gain
        self.declare_parameter('desired_distance', 0.4)
        self.declare_parameter('linear_speed', 3.0)
        self.declare_parameter('kp', 2.0)
        self.desired_distance = self.get_parameter('desired_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.kp = self.get_parameter('kp').value

        # Subscribers & Publishers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
    
    def scan_callback(self, msg: LaserScan):
        # Use 4.71 rad (≈270°) to get the distance on the right side.
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        num_readings = len(msg.ranges)
        
        target_angle = 4.7  # 270° in radians.
        target_index = int((target_angle - angle_min) / angle_inc)
        
        # Average over a small window around target_index.
        window = 5
        indices = range(max(0, target_index - window), min(num_readings, target_index + window + 1))
        valid_readings = [msg.ranges[i] for i in indices if not np.isinf(msg.ranges[i]) and not np.isnan(msg.ranges[i])]
        
        # Check if we have any valid readings:
        if valid_readings:
            avg_distance = sum(valid_readings) / len(valid_readings)
        else:
            self.get_logger().warn("No valid LiDAR readings in the target window; stopping!")
            avg_distance = self.desired_distance  # fallback to desired distance to minimize error
        
        # Compute error (desired minus measured)
        error = self.desired_distance - avg_distance
        angular_z = self.kp * error
        
        # Log for debugging
        self.get_logger().info(f"Avg dist: {avg_distance:.2f}, error: {error:.2f}, angular_z: {angular_z:.2f}")
        
        # Create and publish Twist
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
