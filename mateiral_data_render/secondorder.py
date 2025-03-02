# Copyright 2022 The Kubric Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
# set every random seed
import random
import numpy as np
import os
import sys

random.seed(0)
np.random.seed(0)
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import logging
import os
import time
import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
from material import assign_material_transmission, assign_material_metalic, assign_material_anisotropic, random_assign_material
# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 10)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-2., -2., 0.), (2., 2., 0.)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_static_objects", type=int, default=7,
                    help="minimum number of static (distractor) objects")
parser.add_argument("--max_num_static_objects", type=int, default=15,
                    help="maximum number of static (distractor) objects")
parser.add_argument("--min_num_dynamic_objects", type=int, default=5,
                    help="minimum number of dynamic (tossed) objects")
parser.add_argument("--max_num_dynamic_objects", type=int, default=12,
                    help="maximum number of dynamic (tossed) objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.5)  # 地面摩擦力
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement"],
                    default="fixed_random")  # fix to avoid global flow
parser.add_argument("--max_camera_movement", type=float, default=4.0)
parser.add_argument("--max_motion_blur", type=float, default=0.4)

# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default=r"/home/kubric/kubric/assets/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="/home/kubric/kubric/assets/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="/home/kubric/kubric/assets/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=6, frame_rate=30,
                    resolution=256)
FLAGS = parser.parse_args()
# random select camera
FLAGS.camera = np.random.choice(["fixed_random", "linear_movement"], p=[0.5, 0.5])

start_scenes = 1
num_scenes = 30


# --- Common setups & resources
# create multiple scenes

for scene_id in range(start_scenes, start_scenes + num_scenes):
    # pause the foe 10 seconds
    time.sleep(10)
    FLAGS.job_dir = "scene_{:03d}".format(scene_id)
    print(FLAGS.job_dir)
    scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
    motion_blur = rng.uniform(0, FLAGS.max_motion_blur)
    if motion_blur > 0.0:
        logging.info(f"Using motion blur strength {motion_blur}")

    simulator = PyBullet(scene, scratch_dir)
    renderer = Blender(scene, scratch_dir, adaptive_sampling=False, use_denoising=True, samples_per_pixel=256,
                       motion_blur=motion_blur, background_transparency=False)
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    # --- Populate the scene
    # background HDRI
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    # randomly choose a background
    if np.random.rand() < 0.7:
        FLAGS.backgrounds_split = "train"
    else:
        FLAGS.backgrounds_split = "test"

    if FLAGS.backgrounds_split == "train":
        logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
        hdri_id = rng.choice(train_backgrounds)
    else:
        logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
        hdri_id = rng.choice(test_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    # assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                          friction=1.0,
                          restitution=0.0,
                          static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)


    def get_linear_camera_motion_start_end(
            movement_speed: float,
            inner_radius: float = 8.,
            outer_radius: float = 12.,
            z_offset: float = 0.1,
    ):
        """Sample a linear path which starts and ends within a half-sphere shell."""
        while True:
            camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                         outer_radius,
                                                                         z_offset))
            direction = rng.rand(3) - 0.5
            movement = direction / np.linalg.norm(direction) * movement_speed
            camera_end = camera_start + movement
            if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
                    camera_end[2] > z_offset):
                return camera_start, camera_end


    # Camera
    logging.info("Setting up the Camera...")
    scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
    if FLAGS.camera == "fixed_random":
        scene.camera.position = kb.sample_point_in_half_sphere_shell(
            inner_radius=7., outer_radius=9., offset=0.1)
        scene.camera.look_at((0, 0, 0))
    elif FLAGS.camera == "linear_movement":
        camera_start, camera_end = get_linear_camera_motion_start_end(
            movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement)
        )
        # linearly interpolate the camera position between these two points
        # while keeping it focused on the center of the scene
        # we start one frame early and end one frame late to ensure that
        # forward and backward flow are still consistent for the last and first frames
        for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
            interp = ((frame - FLAGS.frame_start + 1) /
                      (FLAGS.frame_end - FLAGS.frame_start + 3))
            scene.camera.position = (interp * np.array(camera_start) +
                                     (1 - interp) * np.array(camera_end))
            scene.camera.look_at((0, 0, 0))
            scene.camera.keyframe_insert("position", frame)
            scene.camera.keyframe_insert("quaternion", frame)

    # ---- Object placement ----
    train_split, test_split = gso.get_test_split(fraction=0.1)
    #randomly place objects TODO
    if np.random.uniform() < 0.7:
        FLAGS.objects_split = "train"
    else:
        FLAGS.objects_split = "test"

    if FLAGS.objects_split == "train":
        logging.info("Choosing one of the %d training objects...", len(train_split))
        active_split = train_split
    else:
        logging.info("Choosing one of the %d held-out objects...", len(test_split))
        active_split = test_split

    # add STATIC objects
    num_static_objects = rng.randint(FLAGS.min_num_static_objects,
                                     FLAGS.max_num_static_objects + 1)
    logging.info("Randomly placing %d static objects:", num_static_objects)
    for i in range(num_static_objects):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale"] = scale
        scene += obj

        kb.move_until_no_overlap(obj, simulator, spawn_region=STATIC_SPAWN_REGION,
                                 rng=rng)
        obj.friction = 1.0
        obj.restitution = 0.0

        # # TODO here we could adjust the material properties of the static objects
        # obj = random_assign_material(obj)

        obj.metadata["is_dynamic"] = False
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    logging.info("Running 100 frames of simulation to let static objects settle ...")
    _, _ = simulator.run(frame_start=-100, frame_end=0)

    # stop any objects that are still moving and reset friction / restitution
    for obj in scene.foreground_assets:
        if hasattr(obj, "velocity"):
            obj.velocity = (0., 0., 0.)
            obj.friction = 0.5
            obj.restitution = 0.5

    dome.friction = FLAGS.floor_friction
    dome.restitution = FLAGS.floor_restitution

    # Add DYNAMIC objects
    num_dynamic_objects = rng.randint(FLAGS.min_num_dynamic_objects,
                                      FLAGS.max_num_dynamic_objects + 1)
    logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
    for i in range(num_dynamic_objects):
        obj = gso.create(asset_id=rng.choice(active_split))
        assert isinstance(obj, kb.FileBasedObject)
        scale = rng.uniform(0.75, 3.0)
        obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale"] = scale

        # TODO here we could adjust the material properties of the dynamic object
        # obj = random_assign_material(obj)

        scene += obj
        kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION,
                                 rng=rng)
        obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
                        [obj.position[0], obj.position[1], 0])
        obj.metadata["is_dynamic"] = True
        logging.info("    Added %s at %s", obj.asset_id, obj.position)

    if FLAGS.save_state:
        logging.info("Saving the simulator state to '%s' prior to the simulation.",
                     output_dir / "scene.bullet")
        simulator.save_state(output_dir / "scene.bullet")

    # Run dynamic objects simulation
    logging.info("Running the simulation ...")
    animation, collisions = simulator.run(frame_start=0,
                                          frame_end=scene.frame_end + 1)

    # --- Rendering
    if FLAGS.save_state:
        logging.info("Saving the renderer state to '%s' ",
                     output_dir / "scene.blend")
        renderer.save_state(output_dir / "scene.blend")

    logging.info("Rendering the scene ...")
    data_stack = renderer.render(  return_layers = ("rgba", "backward_flow", "forward_flow","segmentation"))

    # sleep 10 seconds
    time.sleep(15)

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                 if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        visible_foreground_assets)
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    kb.write_image_dict(data_stack, output_dir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                      visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(filename=output_dir / "metadata.json", data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=output_dir / "events.json", data={
        "collisions": kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })

    kb.done()
