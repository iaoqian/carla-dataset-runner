from typing import *
import carla
import queue

import numpy as np

from prj_utils import *
from tqdm import tqdm


class CarlaPatch2Prj(CarlaVirtualObject):
    """
    load patch to project in carla world, get point cloud in return value.
    """

    def __init__(self, patch_size):
        super().__init__(patch_size)
        self.patch_data = np.zeros(patch_size)

    def data2pcd(self, depth_data: carla.Image, prj_depth_camera: carla.Sensor):
        patch_p3d, _ = depth_to_local_point_cloud(depth_data, max_depth=0.9)
        c2w_mat = get_camera2world_matrix(prj_depth_camera.get_transform())
        patch_p3d = (c2w_mat @ patch_p3d)[:3]

        return patch_p3d


class PatchSelector:
    def __init__(self, objects_p3d: List):
        self.objects_p3d = objects_p3d

    @staticmethod
    def clustering():
        pass

    @staticmethod
    def _total_variance(p3ds):
        idx = (np.arange(p3ds.shape[1]) + 1) % p3ds.shape[1]
        return ((p3ds[:, idx] - p3ds)[:, :-1] ** 2.0).sum(axis=0).mean().item()

    def __call__(self):
        self.objects_p3d.sort(key=lambda x: self._total_variance(x))
        return [self.objects_p3d[0]]


class PatchProjector:
    def __init__(self, world, patch_size, ego_location):
        self.world = world
        self.patch_size = patch_size
        self._projectors_array = self._get_projectors_array(ego_location)

        self.objects_p3d = self._project_patches()

    def _get_projectors_array(self, ego_location):
        bp_lib = self.world.get_blueprint_library()
        spawn_point = ego_location

        def get_combs(ele_list: list, comb_len=3) -> List[List[int or float]]:
            if comb_len > 1:
                return [[ele] + comb for ele in ele_list for comb in get_combs(ele_list, comb_len - 1)]
            else:
                return [[ele] for ele in ele_list]

        rotations = [0]
        locations = [0]
        fovs = [15]

        projectors = []
        for fov in fovs:
            for rotation in get_combs(rotations, 2):
                for location in get_combs(locations, 3):
                    # no roll
                    rotation = rotation + [0]
                    prj_depth_camera_bp = bp_lib.find('sensor.camera.depth')
                    prj_depth_camera_bp.set_attribute('fov', str(fov))
                    prj_depth_camera_bp.set_attribute('image_size_x', str(self.patch_size[1]))
                    prj_depth_camera_bp.set_attribute('image_size_y', str(self.patch_size[0]))

                    ve_x, ve_y, ve_z = spawn_point.location.x, spawn_point.location.y, spawn_point.location.z + 5
                    ve_ya, ve_pi, ve_ro = spawn_point.rotation.yaw, spawn_point.rotation.pitch, spawn_point.rotation.roll
                    x, y, z = location[0] + ve_x, location[1] + ve_y, location[2] + ve_z
                    yaw, pitch, roll = rotation[0] + ve_ya, rotation[1] + ve_pi, rotation[2] + ve_ro

                    prj_depth_camera_init_trans = \
                        carla.Transform(carla.Location(x=x, y=y, z=z),
                                        carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))

                    prj_depth_camera = self.world.spawn_actor(prj_depth_camera_bp, prj_depth_camera_init_trans,
                                                              attach_to=None,
                                                              attachment_type=carla.AttachmentType.Rigid)

                    prj_depth_image_queue = queue.Queue()
                    prj_depth_camera.listen(prj_depth_image_queue.put)
                    projectors.append({'projector': prj_depth_camera, 'data_queue': prj_depth_image_queue})

        return projectors

    def _project_patches(self):
        objects_p3d = []
        carla_patch = CarlaPatch2Prj(self.patch_size)

        with tqdm(total=len(self._projectors_array), leave=False) as pbar:
            for i, camera in enumerate(self._projectors_array):
                pbar.set_description(f'setting projector {i + 1}')
                prj_depth_image_queue = camera['data_queue']
                prj_depth_camera = camera['projector']

                prj_depth_image = prj_depth_image_queue.get()
                object_p3d = carla_patch.data2pcd(prj_depth_image, prj_depth_camera)
                objects_p3d.append(object_p3d)
                prj_depth_camera.destroy()

                pbar.set_description(f'destroy projector {i + 1}')
                pbar.update(1)

        objects_p3d = PatchSelector(objects_p3d)()

        for i, object_p3d in enumerate(objects_p3d):
            if object_p3d.shape[1] != self.patch_size[0] * self.patch_size[1]:
                print('Value Num Error.')
                _object = np.zeros((3, self.patch_size[0] * self.patch_size[1]))
                object_p3d = object_p3d[:, :min(object_p3d.shape[1], self.patch_size[0] * self.patch_size[1])]
                _object[:, :object_p3d.shape[1]] = object_p3d
                objects_p3d[i] = _object

        return objects_p3d

    def __call__(self, rgb_camera, rgb_image):

        k = numpy.identity(3)
        k[0, 2] = rgb_image.width / 2.0
        k[1, 2] = rgb_image.height / 2.0
        k[0, 0] = k[1, 1] = rgb_image.width / (2.0 * math.tan(rgb_image.fov * math.pi / 360.0))

        world2camera_matrix = get_world2camera_matrix(rgb_camera.get_transform())

        patch_indices = []

        for i, object_p3d in enumerate(self.objects_p3d):
            # print(i)
            p2d = k @ (world2camera_matrix @ np.concatenate([object_p3d, np.ones((1, object_p3d.shape[1]))]))[:3]
            p2d[0] /= p2d[2]
            p2d[1] /= p2d[2]
            # convert to int64 as index
            p2d = numpy.array(p2d[:2] + 0.5, dtype=np.int64)
            mask = (p2d[0] >= 0) & (p2d[1] >= 0) & (p2d[0] < rgb_image.width) & (p2d[1] < rgb_image.height)
            p2d = mask * p2d + ~mask * np.zeros_like(p2d[0])

            c_point = object_p3d[:, object_p3d.shape[1] // 2]
            ray = carla.Location(c_point[0], - c_point[1], c_point[2]) - rgb_camera.get_transform().location
            forward_vec = rgb_camera.get_transform().get_forward_vector()
            # to make sure patch is in the front of vehicle and display it.
            if forward_vec.dot(ray) > 0:
                # replace pixels in camera output with patch pixels
                # rgb_image[-p2d[1], -p2d[0], :3] = np.array(object_color * 255, dtype=np.uint8)
                patch_indices.append(
                    np.stack((
                        (rgb_image.height - p2d[1] - 1).reshape(self.patch_size),
                        (rgb_image.width - p2d[0] - 1).reshape(self.patch_size),), axis=0)
                )
            else:
                patch_indices.append(np.zeros((2,) + self.patch_size))

        return np.stack(patch_indices, axis=0).astype(np.int64)