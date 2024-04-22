import blenderproc as bproc
import numpy as np
import os
import time
import glob

DATA_PATH = "/dl/volatile/students/projects/llms_for_code/master_thesis/"
MODELS_PATH = os.path.join(DATA_PATH, "models")
num_scenes = 700


# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
    
def split_plys(all_plys):
    np.random.seed(8)
    min_chunk_size = 6
    max_chunk_size = 13
    total_elements = len(all_plys)
    chunk_sizes = []
    
    while total_elements > 0:
        chunk_size = np.random.randint(min_chunk_size, max_chunk_size + 1)
        chunk_size = min(chunk_size, total_elements) 
        chunk_sizes.append(chunk_size)
        total_elements -= chunk_size
    chunks = np.split(all_plys, np.cumsum(chunk_sizes)[:-1])
    chunks_array = np.array(chunks)
    return chunks_array



def render():
    all_plys = glob.glob(os.path.join(MODELS_PATH, '*.ply'))
    samples = split_plys(all_plys)

    # initialize room
    bproc.init()
    # Setup room
    room_planes = [
        bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
        bproc.object.create_primitive(
            'PLANE',
            scale=[2, 2, 1],
            location=[0, -2, 2],
            rotation=[-1.570796, 0, 0],
        ),
        bproc.object.create_primitive(
            'PLANE',
            scale=[2, 2, 1],
            location=[0, 2, 2],
            rotation=[1.570796, 0, 0],
        ),
        bproc.object.create_primitive(
            'PLANE',
            scale=[2, 2, 1],
            location=[2, 0, 2],
            rotation=[0, -1.570796, 0],
        ),
        bproc.object.create_primitive(
            'PLANE',
            scale=[2, 2, 1],
            location=[-2, 0, 2],
            rotation=[0, 1.570796, 0],
        ),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape='BOX',
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )
    # Setup lighting
    light_plane = bproc.object.create_primitive(
        'PLANE', scale=[3, 3, 1], location=[0, 0, 10]
    )
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')
    light_point = bproc.types.Light()
    light_point.set_energy(200)
    # Load textures
    cc_textures = bproc.loader.load_ccmaterials(
        "/mvtec/home/nincicn/work/code/BlenderProc/ressources/cc_textures"
    )

    # # Set camera params and enable depth rendering
    fx, fy        = (759.4112396240234, 759.4111490951599)
    width, height = (224,224)
    cx, cy        = width / 2, height / 2
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    bproc.python.camera.CameraUtility.set_intrinsics_from_K_matrix(K, width, height)
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

    # create scenes
    objs = []
    t0 = time.time()
    for i, sampled_plys in enumerate(samples):
        num_poses = sampled_plys.shape[0] * 4
        # reset scene
        bproc.object.delete_multiple(objs)
        bproc.python.utility.Utility.reset_keyframes()
        # Load objects into the scene
        objs = []
        for ply in sampled_plys:
            obj = bproc.loader.load_obj(os.path.join(MODELS_PATH, ply))[0]
            # obj.replace_materials(np.random.choice(cc_textures))
            obj.enable_rigidbody(
                True, friction=100.0, linear_damping=0.99, angular_damping=0.99
            )
            obj_id = int(ply.split("obj_")[-1].removesuffix(".ply"))
            obj.set_cp("category_id", obj_id)
            objs.append(obj)

        # sample light color and strenght from ceiling
        light_plane_material.make_emissive(
            emission_strength=np.random.uniform(3, 6),
            emission_color=np.random.uniform(
                [0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]
            ),
        )
        light_plane.replace_materials(light_plane_material)

        # sample point light on shell
        light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
        location = bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=1,
            radius_max=1.5,
            elevation_min=5,
            elevation_max=89,
            uniform_volume=False,
        )
        light_point.set_location(location)

        # sample CC Texture and assign to room planes
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

        # Sample object poses and check collisions
        bproc.object.sample_poses(
            objects_to_sample=objs,
            sample_pose_func=sample_pose_func,
            objects_to_check_collisions=objs,
            max_tries=1000,
        )

        # Physics Positioning
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=10,
            check_object_interval=1,
            substeps_per_frame=20,
            solver_iters=25,
        )

        # BVH tree used for camera obstacle checks
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

        poses = 0
        while poses < num_poses:
            # Sample location
            location = bproc.sampler.shell(
                center=[0, 0, 0],
                radius_min=0.61,
                radius_max=1.24,
                elevation_min=5,
                elevation_max=89,
                uniform_volume=False,
            )
            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            poi = bproc.object.compute_poi(np.random.choice(objs, size=3))
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854)
            )
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(
                location, rotation_matrix
            )

            # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
            if bproc.camera.perform_obstacle_in_view_check(
                cam2world_matrix, {"min": 0.3}, bop_bvh_tree
            ):
                # Persist camera pose
                bproc.camera.add_camera_pose(cam2world_matrix)
                poses += 1

        # Render the scene
        data = bproc.renderer.render()

        try:
            bproc.writer.write_bop(
                os.path.join(DATA_PATH, 'bop_data'),
                target_objects=objs,
                colors=data["colors"],
                depths=data["depth"],
                color_file_format="JPEG",
                ignore_dist_thres=10,
            )
        except:
            continue
    print(f"Render time per blender scene:{(time.time() - t0)/num_scenes:.2f}s\nRender time per camera pose: {(time.time() - t0)/(num_scenes*num_poses):.2f}s")


if __name__ == "__main__":
    render()
