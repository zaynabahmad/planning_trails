import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import numpy as np

class OptimizedGapFollower(Node):
    def __init__(self):
        super().__init__('optimized_gap_follower')
        
        # Parameters
        self.max_speed = 2.0
        self.max_angle_vel = 1.5
        self.r_b = 0.2  # Safety bubble (inflation) radius in meters
        self.t = 1.5    # Threshold distance: any reading below this is considered a near obstacle
        self.max_range = 7.0  # Maximum sensor reading (used to replace inf values)
        
        self.robot_yaw = 0.0          # Current yaw (from odometry)
        self.latest_heading_error = 0.0
        
        # For gradual angular velocity update:
        self.current_angular_vel = 0.0  # Current commanded angular velocity
        self.angular_step = 0.6         # Step size for changing angular velocity
        
        # Subscribers & Publishers
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 1)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 1)
    
    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        # Compute yaw from quaternion (only yaw needed for 2D navigation)
        _, _, self.robot_yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.get_logger().info(f"Current robot yaw: {self.robot_yaw:.2f}")

    def euler_from_quaternion(self, x, y, z, w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return 0.0, 0.0, yaw

    def lidar_callback(self, scan_msg):
        """
        Process LiDAR data by restricting attention to the front field of view, marking any beam
        that is too close to an obstacle (and a safety window around it) as invalid, and then selecting
        the beam with the maximum original range among those with distance < 6.0 meters.
        """
        # Get LiDAR ranges and replace infinite values with max_range.
        ranges = np.array(scan_msg.ranges)
        ranges = np.where(np.isinf(ranges), self.max_range, ranges)
        
        # Since the LiDAR measures 360° starting at 0 (front) and increasing counterclockwise,
        # we consider only the front field of view (0° to 90° and 270° to 360°).
        total_beams = len(ranges)
        degrees_per_beam = 360.0 / total_beams
        
        # Indices for 0° to 90°
        index_90 = int(90 / degrees_per_beam)
        # Indices for 270° to 360°
        index_270 = int(270 / degrees_per_beam)
        
        # Create a boolean mask: set True for beams in the front FOV.
        front_mask = np.zeros_like(ranges, dtype=bool)
        front_mask[0:index_90 + 1] = True
        front_mask[index_270:total_beams] = True
        
        # Select only the front beams.
        ranges_front = ranges[front_mask]
        
        # Copy original front ranges for inflation processing.
        inflated_ranges = ranges_front.copy()
        angle_inc = scan_msg.angle_increment  # Angular increment in radians

        # For every beam in the front FOV with a range below the safety threshold,
        # zero out a window around it.
        for i, r in enumerate(ranges_front):
            if r < self.t:
                # Compute the inflation angle: the safety bubble subtends an angle given by arcsin(r_b / r).
                # If the reading is extremely close (r < r_b), use a wide window.
                if r > self.r_b:
                    inflation_angle = np.arcsin(self.r_b / r)
                else:
                    inflation_angle = np.pi / 2  # maximum inflation if too close
                window = int(np.ceil(inflation_angle / angle_inc))
                start = max(0, i - window)
                end = min(len(ranges_front) - 1, i + window)
                inflated_ranges[start:end+1] = 0.0

        # Consider all beams with non-zero inflated range as valid.
        valid_indices = np.where(inflated_ranges > 0)[0]
        if valid_indices.size == 0:
            self.get_logger().warn("No valid path found! Stopping robot.")
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0
            self.cmd_pub.publish(twist_msg)
            return

        # Filter valid beams with a distance less than 6.0 meters.
        valid_less_than_6 = valid_indices[ranges_front[valid_indices] < 7.0]
        if valid_less_than_6.size > 0:
            best_index = valid_less_than_6[np.argmax(ranges_front[valid_less_than_6])]
        else:
            # Fallback: choose the beam with the maximum original range among all valid beams.
            best_index = valid_indices[np.argmax(ranges_front[valid_indices])]
        
        # Map the best_index (from ranges_front) back to the original indices.
        original_indices = np.where(front_mask)[0]
        absolute_best_index = original_indices[best_index]
        
        # Compute the best heading in radians.
        best_heading = scan_msg.angle_min + absolute_best_index * angle_inc
        selected_distance = ranges_front[best_index]
        self.get_logger().info(
            f"Selected beam index: {absolute_best_index}, heading: {best_heading:.2f} rad, "
            f"distance: {selected_distance:.2f} m"
        )
        
        # Compute heading error (assuming LiDAR's angle_min aligns with the robot’s forward direction).
        heading_error = best_heading - 0.01  # applying a small offset (noise correction)
        self.latest_heading_error = heading_error

        # Adjust linear speed: reduce speed on sharper turns.
        if abs(heading_error) > 0.3:
            linear_speed = self.max_speed * 0.7
        else:
            linear_speed = self.max_speed

        self.update_robot_movement(linear_speed)

    def steering_control(self, heading_error, k_p=0.7):
        """
        Compute angular velocity from heading error using a proportional controller,
        ensuring that angles in [π, 2π] become negative (so the robot turns to the right).
        """
        # Normalize heading_error to [-pi, pi].
        if heading_error > math.pi:
            heading_error -= 2.0 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2.0 * math.pi
        
        # Compute the proportional steering command.
        angular_vel = k_p * heading_error
        
        # Clip the angular velocity to the maximum allowed value.
        angular_vel = max(min(angular_vel, self.max_angle_vel), -self.max_angle_vel)
        
        return angular_vel

    
    def update_robot_movement(self, linear_speed):
        """
        Gradually update the angular velocity toward the target while publishing the command.
        """
        target_angular_vel = self.steering_control(self.latest_heading_error)
        # Smooth the angular velocity update for stability at high speeds.
        if self.current_angular_vel < target_angular_vel:
            self.current_angular_vel = min(self.current_angular_vel + self.angular_step, target_angular_vel)
        elif self.current_angular_vel > target_angular_vel:
            self.current_angular_vel = max(self.current_angular_vel - self.angular_step, target_angular_vel)
        
        twist_msg = Twist()
        twist_msg.linear.x = linear_speed
        twist_msg.angular.z = self.current_angular_vel
        self.cmd_pub.publish(twist_msg)
        self.get_logger().info(
            f"Command: linear.x={linear_speed:.2f}, angular.z={self.current_angular_vel:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedGapFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
