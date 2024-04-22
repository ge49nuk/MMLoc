import objaverse
import multiprocessing
import numpy as np
import torch
import trimesh
import os
from pointnet2_ops import pointnet2_utils

DATA_PATH = "/dl/volatile/students/projects/llms_for_code/master_thesis/"
MODELS_PATH = os.path.join(DATA_PATH, "models")
MESH_DATA_PATH = os.path.join(DATA_PATH, "mesh_data/")
os.environ['PYOPENGL_PLATFORM'] = 'egl'
objaverse._VERSIONED_PATH = os.path.join(DATA_PATH, "objaverse")

num_meshes = 7000

def point_cloud_to_voxel_grid(point_cloud, grid_size=256):
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    dimensions = max_coords - min_coords
    voxel_size = max(dimensions / grid_size) + 1e-5
    voxel_grid_shape = (grid_size, grid_size, grid_size)
    voxel_grid = np.zeros(voxel_grid_shape, dtype=bool)
    voxel_indices = ((point_cloud - min_coords) / voxel_size).astype(int)
    for idx in voxel_indices:
        voxel_grid[tuple(idx)] = True
    return voxel_grid

def fps(data, number):
    data = data.unsqueeze(0)
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data[0]

def main():
    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)
    if not os.path.exists(MESH_DATA_PATH):
        os.makedirs(MESH_DATA_PATH)
    seen_uids = []
    next_id = 0
    if os.path.exists(os.path.join(MODELS_PATH, "seen_uids.pt")):
        seen_uids = torch.load(os.path.join(MODELS_PATH, "seen_uids.pt"))
        next_id = len(seen_uids)

    # Get random UIDS and download their corresponding objects
    uids_all = np.array(objaverse.load_uids())
    uids = np.random.choice(uids_all, num_meshes)
    annotations = objaverse.load_annotations(uids)
    cc_by_uids = [
        uid
        for uid, annotation in annotations.items()
        if annotation["license"] == "by" and annotation["faceCount"] < 100000
    ]
    processes = multiprocessing.cpu_count()
    objects = objaverse.load_objects(
        uids=cc_by_uids, download_processes=processes
    )
    for uid, path in objects.items():
        if uid in seen_uids:
            continue
        try:
            obj = trimesh.load(path)
            mesh = trimesh.util.concatenate(obj.geometry.values())
            if not (
                type(mesh) == trimesh.base.Trimesh
                and type(mesh.visual) == trimesh.visual.texture.TextureVisuals
                and mesh.visual.defined
            ):
                continue

            # Normalize mesh
            bbox_min, bbox_max = mesh.bounds
            bbox_center = (bbox_min + bbox_max) / 2.0
            max_dimension = max(bbox_max - bbox_min)
            scale_factor = 0.2 / max_dimension
            mesh.vertices -= bbox_center
            mesh.vertices *= scale_factor

            # Save data in dict
            vertices = mesh.vertices
            # colors =  mesh.visual.vertex_colors
            # voxels = point_cloud_to_voxel_grid(vertices)
            # voxels_tensor = torch.tensor(voxels, dtype=torch.bool)
            vertices = fps(torch.tensor(vertices, dtype=torch.float32).cuda(), 1024).cpu()
            
            # colors_tensor = torch.tensor(colors)
            data_dict = {
                "obj_id": next_id,
                "vertices": vertices,
                # "colors": colors_tensor,
            }

            # Save mesh and mesh data
            mesh.export(os.path.join(MODELS_PATH, f"obj_{str(next_id).zfill(6)}.ply"))
            torch.save(data_dict, os.path.join(MESH_DATA_PATH, f"obj_{str(next_id).zfill(6)}.pt"))
            seen_uids.append(uid)
            with open(os.path.join(MODELS_PATH, "id_to_uid.txt"), "a+") as f:
                f.write(str(next_id)+":"+uid+"\n")
            next_id += 1

        except Exception as e:
            print(e)
            continue
    torch.save(seen_uids, os.path.join(MODELS_PATH, "seen_uids.pt"))

if __name__ == "__main__":
    main()
