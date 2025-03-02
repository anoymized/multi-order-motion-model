import kubric as kb
import math
import numpy as np


def get_linear_camera_motion_start_end(
        movement_speed: float,
        inner_radius: float = 8.,
        outer_radius: float = 20.,
        z_offset: float = [0.1, 2],
):
    """Sample a linear path which starts and ends within a half-sphere shell."""
    while True:
        z  = np.random.uniform(low=z_offset[0], high=z_offset[1])
        camera_start = np.array(
            kb.sample_point_in_half_sphere_shell(inner_radius, outer_radius, z)
        )
        # 原来是: direction = rng.rand(3) - 0.5
        direction = np.random.rand(3) - 0.5


        movement = direction / np.linalg.norm(direction) * movement_speed
        camera_end = camera_start + movement

        if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                camera_end[2] > z):
            return camera_start, camera_end


def get_arc_path_points(
        inner_radius=7., outer_radius=9., z_offset=[0.15, 2],
        arc_angle_range=(math.pi / 6, math.pi / 2),  # 弧线跨度范围 [30°, 90°]
        num_points=48
):
    """在半球壳内采样一个圆弧的起始角度和结束角度，并返回对应的轨迹点。
       可以根据需要修改采样方式和弧线形状。
    """
    # 1. 在内外半径之间随机一个半径
    radius = np.random.uniform(low=inner_radius, high=outer_radius)

    # 2. 随机一个起始角度 start_angle
    start_angle = np.random.uniform(low=0, high=2 * math.pi)
    # 3. 在给定 arc_angle_range 范围内随机得到弧线跨度
    arc_span = np.random.uniform(low=arc_angle_range[0], high=arc_angle_range[1])
    # 4. 结束角度
    end_angle = start_angle + arc_span

    # 将弧线分为若干段，返回每段对应的 (x, y, z)

    angles = np.linspace(start_angle, end_angle, num_points)
    z_offset_start = np.random.uniform(low=z_offset[0], high=z_offset[1])
    z_offset_end = np.random.uniform(low=z_offset[0], high=z_offset[1])
    z_interp = np.linspace(z_offset_start, z_offset_end, num_points)

    # x(t), y(t) 沿水平圆弧，z(t) 固定或带小偏移
    path_points = []
    for i, theta in enumerate(angles):
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = z_interp[i]
        path_points.append((x, y, z))

    return path_points