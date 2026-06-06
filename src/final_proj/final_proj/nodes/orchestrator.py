import os
import math
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

from final_proj.environment.map_loader import MapLoader
from final_proj.planning.astar import astar
from final_proj.memory.skill_store import SkillStore

try:
    from final_proj.planning.replanner import Replanner
    from final_proj.llm.client import LLMClient
    from final_proj.llm.prompt_builder import PromptBuilder
    from final_proj.llm.response_parser import ResponseParser
    from final_proj.environment.context_extractor import ContextExtractor
except Exception:
    Replanner = None


class Orchestrator(Node):
    def __init__(self):
        super().__init__('orchestrator')

        # parameters
        self.declare_parameter('MAP_YAML_PATH', '')
        self.declare_parameter('map_yaml', '')
        # default goal as a double array to avoid parameter type mismatch
        self.declare_parameter('goal', [0.0, 0.0])

        # hardcode map path to repository data/map.yaml by default
        pkg_root = Path(__file__).resolve().parents[1]
        hardcoded_map = str(pkg_root / 'data' / 'map.yaml')
        map_path = hardcoded_map
        self.get_logger().info(f'Hardcoded MAP path: {map_path}')

        self.map_loader = MapLoader(map_path)
        self.grid = self.map_loader.get_grid()
        self.resolution = self.map_loader.get_resolution()
        self.origin = self.map_loader.get_origin()

        # skill store
        pkg_root = Path(__file__).resolve().parents[1]
        skills_path = pkg_root / 'data' / 'skills.json'
        self.skill_store = SkillStore(str(skills_path))

        # optional replanner
        self.replanner = None
        if os.environ.get('GROQ_API_KEY') and Replanner is not None:
            try:
                client = LLMClient()
                builder = PromptBuilder()
                parser = ResponseParser()
                extractor = ContextExtractor()
                self.replanner = Replanner(builder, parser, client, extractor)
                self.get_logger().info('LLM Replanner enabled')
            except Exception as e:
                self.get_logger().warn(f'Failed to enable LLM Replanner: {e}')

        # ros interfaces
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.latest_odom = None

        self.timer = self.create_timer(1.0, self.timer_callback)

        self.get_logger().info('Orchestrator started')

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def world_to_grid(self, x, y):
        ox, oy = self.origin[0], self.origin[1]
        gx = int((x - ox) / self.resolution)
        gy = int((y - oy) / self.resolution)
        # map_loader stores flipped grid (y upwards), so map y index must be converted
        height, width = self.grid.shape
        gy = height - 1 - gy
        return (gx, gy)

    def grid_to_world(self, gx, gy):
        # convert back accounting for flipped grid
        height, width = self.grid.shape
        gy_conv = height - 1 - gy
        ox, oy = self.origin[0], self.origin[1]
        x = gx * self.resolution + ox + self.resolution / 2.0
        y = gy_conv * self.resolution + oy + self.resolution / 2.0
        return x, y

    def make_cost_map(self):
        # grid: 1=occupied, 0=free, -1=unknown
        cost_map = np.full(self.grid.shape, np.inf, dtype=float)
        cost_map[self.grid == 0] = 1.0
        cost_map[self.grid == -1] = 5.0
        return cost_map

    def timer_callback(self):
        if self.latest_odom is None:
            return

        goal_param = self.get_parameter('goal').value

        # normalize goal parameter to a list of two floats
        goal = None
        if isinstance(goal_param, str):
            try:
                goal = json.loads(goal_param)
            except Exception:
                s = goal_param.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1]
                parts = [p.strip() for p in s.split(',') if p.strip()]
                try:
                    goal = [float(p) for p in parts]
                except Exception:
                    goal = None
        elif isinstance(goal_param, (list, tuple)):
            goal = list(goal_param)
        elif isinstance(goal_param, bytes):
            try:
                decoded = goal_param.decode('utf-8')
                goal = json.loads(decoded)
            except Exception:
                goal = None

        if not goal or len(goal) < 2:
            # nothing to do without a valid goal
            return

        try:
            gx = float(goal[0])
            gy = float(goal[1])
        except Exception:
            self.get_logger().warn('Invalid goal parameter; expected [x,y]')
            return

        start_x = self.latest_odom.pose.pose.position.x
        start_y = self.latest_odom.pose.pose.position.y

        start_cell = self.world_to_grid(start_x, start_y)
        goal_cell = self.world_to_grid(gx, gy)

        cost_map = self.make_cost_map()
        path = astar(cost_map, start_cell, goal_cell)

        if path is None and self.replanner is not None:
            # attempt replanning using LLM if available
            grid = self.grid.copy()
            robot_pose = {'x': start_x, 'y': start_y}
            modified_grid, decision = self.replanner.replan(grid, start_cell, goal_cell, [], robot_pose)
            if modified_grid is not None:
                self.grid = modified_grid
                cost_map = self.make_cost_map()
                path = astar(cost_map, start_cell, goal_cell)

        if path is None:
            self.get_logger().info('No path found')
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return

        # follow first step
        next_cell = path[1] if len(path) > 1 else path[0]
        wx, wy = self.grid_to_world(next_cell[0], next_cell[1])

        dx = wx - start_x
        dy = wy - start_y
        yaw = self.quaternion_to_yaw(
            self.latest_odom.pose.pose.orientation.x,
            self.latest_odom.pose.pose.orientation.y,
            self.latest_odom.pose.pose.orientation.z,
            self.latest_odom.pose.pose.orientation.w,
        )

        target_yaw = math.atan2(dy, dx)
        yaw_err = target_yaw - yaw
        # normalize
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))

        cmd = Twist()
        # simple controller
        cmd.linear.x = max(0.0, 0.5 * math.hypot(dx, dy)) if abs(yaw_err) < 0.5 else 0.0
        cmd.angular.z = 1.0 * yaw_err
        self.cmd_pub.publish(cmd)

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
