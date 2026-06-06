# load corrected map, run astar, follow path with tighter motion control, then test corruption

import math
import time
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from final_proj.environment.map_loader import MapLoader
from final_proj.environment.corruption import Corruptor
from final_proj.planning.astar import astar


def yaw_from_quaternion(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def neighbors(cell, grid):
    x, y = cell
    height, width = grid.shape
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == 0:
            yield (nx, ny)


def bfs_reachable(grid, start):
    q = deque([start])
    visited = {start}
    while q:
        current = q.popleft()
        for nb in neighbors(current, grid):
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    return visited


def farthest_cell(start, reachable):
    sx, sy = start
    best = None
    best_dist = -1
    for cell in reachable:
        x, y = cell
        dist = abs(x - sx) + abs(y - sy)
        if dist > best_dist:
            best_dist = dist
            best = cell
    return best


class NavigationTester(Node):
    def __init__(self):
        super().__init__("navigation_tester")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pose_sub = self.create_subscription(Odometry, "/ground_truth", self.pose_callback, 10)
        self.pose_ready = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def pose_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.pose_ready = True

    def stop(self):
        msg = Twist()
        self.cmd_pub.publish(msg)

    def wait_for_pose(self):
        while rclpy.ok() and not self.pose_ready:
            rclpy.spin_once(self, timeout_sec=0.1)

    def publish_for_duration(self, linear_x, angular_z, duration):
        end_time = time.time() + duration
        while rclpy.ok() and time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.01)
            msg = Twist()
            msg.linear.x = linear_x
            msg.angular.z = angular_z
            self.cmd_pub.publish(msg)
            time.sleep(0.05)
        self.stop()
        time.sleep(0.2)
        rclpy.spin_once(self, timeout_sec=0.05)

    def turn_and_go(self, dx, dy):
        target_angle = math.atan2(dy, dx)
        error = target_angle - self.yaw
        error = math.atan2(math.sin(error), math.cos(error))

        while abs(error) > 0.08 and rclpy.ok():
            direction = 0.18 if error > 0 else -0.18
            self.publish_for_duration(0.0, direction, 0.2)
            error = target_angle - self.yaw
            error = math.atan2(math.sin(error), math.cos(error))

        distance = math.sqrt(dx * dx + dy * dy)
        moved = 0.0
        step = 0.08

        while moved < distance - 0.05 and rclpy.ok():
            self.publish_for_duration(0.08, 0.0, 0.35)
            moved += step

    def grid_to_world(self, cell, resolution, origin):
        gx, gy = cell
        ox, oy, _ = origin
        wx = ox + (gx + 0.5) * resolution
        wy = oy + (gy + 0.5) * resolution
        return wx, wy

    def world_to_grid(self, x, y, resolution, origin, grid_shape):
        ox, oy, _ = origin
        gx = int((x - ox) / resolution)
        gy = int((y - oy) / resolution)
        height, width = grid_shape
        if gx < 0 or gy < 0 or gx >= width or gy >= height:
            return None
        return (gx, gy)

    def nearest_free_cell(self, grid, center, search_radius=30):
        cx, cy = center
        height, width = grid.shape
        best = None
        best_dist = 10**9
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                x = cx + dx
                y = cy + dy
                if 0 <= x < width and 0 <= y < height and grid[y, x] == 0:
                    dist = abs(dx) + abs(dy)
                    if dist < best_dist:
                        best_dist = dist
                        best = (x, y)
        return best

    def follow_path(self, path, resolution, origin):
        self.wait_for_pose()
        for i in range(1, len(path)):
            wx, wy = self.grid_to_world(path[i], resolution, origin)
            dx = wx - self.x
            dy = wy - self.y
            self.turn_and_go(dx, dy)
            self.stop()
            time.sleep(0.15)
        self.stop()


def main():
    rclpy.init()
    node = NavigationTester()

    map_path = Path("/root/ros2_ws/src/pa3/maze.yml")
    loader = MapLoader(map_path)
    grid = loader.load()

    print("Map loaded")
    print("Grid shape:", grid.shape)
    print("Resolution:", loader.get_resolution())
    print("Origin:", loader.get_origin())

    node.wait_for_pose()

    robot_cell = node.world_to_grid(node.x, node.y, loader.get_resolution(), loader.get_origin(), grid.shape)
    if robot_cell is None:
        print("Robot outside map")
        node.destroy_node()
        rclpy.shutdown()
        return

    start = node.nearest_free_cell(grid, robot_cell, search_radius=30)
    if start is None:
        print("No nearby free start cell found")
        node.destroy_node()
        rclpy.shutdown()
        return

    reachable = bfs_reachable(grid, start)
    goal = farthest_cell(start, reachable)

    if goal is None or goal == start:
        print("No valid goal found")
        node.destroy_node()
        rclpy.shutdown()
        return

    print("Robot pose:", (node.x, node.y, node.yaw))
    print("Robot cell:", robot_cell)
    print("Start:", start)
    print("Goal:", goal)
    print("Reachable cells:", len(reachable))

    path_clean = astar(grid, start, goal, allow_unknown=False)
    if path_clean is None:
        print("No path found on clean map")
    else:
        print("Clean map path length:", len(path_clean))
        print("Clean path:", path_clean)
        node.follow_path(path_clean, loader.get_resolution(), loader.get_origin())
        time.sleep(1.0)
        node.stop()

    corruptor = Corruptor(corruption_rate=0.1)
    corrupted_grid = corruptor.inject_random_corruption(grid)

    path_corrupted = astar(corrupted_grid, start, goal, allow_unknown=False)
    if path_corrupted is None:
        print("No path found on corrupted map")
    else:
        print("Corrupted map path length:", len(path_corrupted))
        print("Corrupted path:", path_corrupted)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()