import os
import logging
import numpy as np
import bpy

import kubric as kb
from kubric.renderer import Blender






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
parser.set_defaults(
    frame_end=5,
    resolution=(512, 512),
)
FLAGS = parser.parse_args()



# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir,
                   samples_per_pixel=256,
                   background_transparency=False)

# --- Fetch shapenet
hdri_id = 'bush_restaurant'
source_path = os.getenv("SHAPENET_GCP_BUCKET", r"/home/kubric/kubric/assets/ShapeNetCore.v2.json")
shapenet = kb.AssetSource.from_manifest(source_path)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
background_hdri = hdri_source.create(asset_id=hdri_id)
renderer._set_ambient_light_hdri(background_hdri.filename)

scene.metadata["background"] = hdri_id

kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
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



# --- Add Klevr-like lights to the scene
scene += kb.assets.utils.get_clevr_lights(rng=rng)
scene.ambient_illumination = kb.Color(0.1, 0.1, 0.1)

# --- Add shadow-catcher floor
floor = kb.Cube(name="floor", scale=(100, 100, 1), position=(0, 0, -1))
scene += floor
# Make the floor transparent except for catching shadows
# Together with background_transparency=True (above) this results in
# the background being transparent except for the object shadows.
if bpy.app.version > (3, 0, 0):
  floor.linked_objects[renderer].is_shadow_catcher = True
else:
  floor.linked_objects[renderer].cycles.is_shadow_catcher = True

# --- Keyframe the camera
scene.camera = kb.PerspectiveCamera()
for frame in range(FLAGS.frame_start, FLAGS.frame_end + 1):
  scene.camera.position = kb.sample_point_in_half_sphere_shell(1.5, 1.7, 0.1)
  scene.camera.look_at((0, 0, 0))
  scene.camera.keyframe_insert("position", frame)
  scene.camera.keyframe_insert("quaternion", frame)

# --- Fetch a random (airplane) asset
airplane_ids = [name for name, spec in shapenet._assets.items()
                if spec["metadata"]["category"] == "airplane"]
for name, spec in shapenet._assets.items():
    print(name, spec["metadata"]["category"])

asset_id = rng.choice(airplane_ids) #< e.g. 02691156_10155655850468db78d106ce0a280f87
obj = shapenet.create(asset_id=asset_id)
logging.info(f"selected '{asset_id}'")

# --- make object flat on X/Y and not penetrate floor
obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)
obj.position = obj.position - (0, 0, obj.aabbox[0][2])
# add your favorite material
obj.material = kb.PrincipledBSDFMaterial(name="material")
obj.material.transmission = 1.0
obj.material.roughness = 0
obj.material.transmission_roughness = 0
obj.material.color = kb.Color(0.5, 0.5, 0.5)

scene.add(obj)

# --- Rendering
logging.info("Rendering the scene ...")
renderer.save_state(output_dir / "scene.blend")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    [obj]).astype(np.uint8)

kb.file_io.write_rgba_batch(data_stack["rgba"], output_dir)
# kb.file_io.write_depth_batch(data_stack["depth"], output_dir)
# kb.file_io.write_segmentation_batch(data_stack["segmentation"], output_dir)

# --- Collect metadata
logging.info("Collecting and storing metadata for each object.")
data = {
  "metadata": kb.get_scene_metadata(scene),
  "camera": kb.get_camera_info(scene.camera),
  "object": kb.get_instance_info(scene, [obj])
}
kb.file_io.write_json(filename=output_dir / "metadata.json", data=data)
kb.done()
