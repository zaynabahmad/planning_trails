import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')
        # Declare parameters for tuning the behavior
        self.declare_parameter('desired_distance', 0.5)
        self.declare_parameter('linear_speed', 1.3)
        self.declare_parameter('kp', 4.0)
        self.declare_parameter('max_angular_speed', 2.0)
        self.declare_parameter('window_size', 40)  # number of indices on each side to consider
        self.declare_parameter('safety_distance', 0.3)  # for front obstacle detection

        self.desired_distance = self.get_parameter('desired_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.kp = self.get_parameter('kp').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.window_size = self.get_parameter('window_size').value
        self.safety_distance = self.get_parameter('safety_distance').value

        # Subscribers & Publishers
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
    
    def scan_callback(self, msg: LaserScan):
        # Calculate the index corresponding to the right side (approx. 270° or 4.71 rad)
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        num_readings = len(msg.ranges)
        
        target_angle = 4.71  # roughly 270° in radians
        target_index = int((target_angle - angle_min) / angle_inc)
        
        # Define the window indices around the target index
        start_index = max(0, target_index - self.window_size)
        end_index = min(num_readings, target_index + self.window_size + 1)
        valid_readings = [
            msg.ranges[i] for i in range(start_index, end_index)
            if not np.isinf(msg.ranges[i]) and not np.isnan(msg.ranges[i])
        ]
        
        if valid_readings:
            # Compute error for each valid reading: desired_distance - actual reading.
            errors = [self.desired_distance - r for r in valid_readings]
            avg_error = np.mean(errors)
            mean_distance = np.mean(valid_readings)
        else:
            self.get_logger().warn("No valid LiDAR readings on right side. Stopping!")
            return

        # Proportional control for angular velocity based on the average error
        angular_z = self.kp * avg_error
        angular_z = max(min(angular_z, self.max_angular_speed), -self.max_angular_speed)
        
        # Adjust forward speed based on the magnitude of the average error
        adjusted_linear_speed = self.linear_speed * max(0.5, 1 - abs(avg_error))
        
        # Check for obstacles directly in front of the robot
        front_sector_deg = 10  # degrees on each side of the forward direction
        front_sector_rad = np.deg2rad(front_sector_deg)
        front_index_range = int(front_sector_rad / angle_inc)
        center_index = int((0 - angle_min) / angle_inc)
        
        front_readings = [
            msg.ranges[i] for i in range(max(0, center_index - front_index_range),
                                         min(num_readings, center_index + front_index_range + 1))
            if not np.isinf(msg.ranges[i]) and not np.isnan(msg.ranges[i])
        ]
        if front_readings and min(front_readings) < self.safety_distance:
            self.get_logger().warn("Obstacle detected in front! Stopping forward motion.")
            adjusted_linear_speed = 0.0

        self.get_logger().info(
            f"Mean distance: {mean_distance:.2f}, Avg error: {avg_error:.2f}, "
            f"angular_z: {angular_z:.2f}, linear: {adjusted_linear_speed:.2f}"
        )
        
        # Publish the computed Twist message
        twist = Twist()
        twist.linear.x = adjusted_linear_speed
        twist.angular.z = angular_z
        self.cmd_pub.publish(twist)
    
    def publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()  # Ensure the robot stops before shutdown.
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
