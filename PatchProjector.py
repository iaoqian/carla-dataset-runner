from typing import *
import carla
import queue

from prj_utils import *


class CarlaPatch2Prj(CarlaVirtualObject):
    """
    load patch to project in carla world, get point cloud in return value.
    """
    def __init__(self, data):
        super().__init__(data)
        patch_array = np.load(data)
        self.patch_data = patch_array

    def data2pcd(self, depth_data: carla.Image, prj_depth_camera: carla.Sensor):
        patch_p3d, patch_color = depth_to_local_point_cloud(depth_data, self.patch_data, max_depth=0.9)
        c2w_mat = get_camera2world_matrix(prj_depth_camera.get_transform())
        patch_p3d = (c2w_mat @ patch_p3d)[:3]

        return patch_p3d, patch_color


class PatchSelector:
    def __init__(self, objects_p3d: List, objects_colors: List):
        self.objects_p3d = objects_p3d
        self.objects_colors = objects_colors

    @staticmethod
    def clustering():
        pass

    def __call__(self):
        pass

class PatchProjector:
    def __init__(self, world, patch_size, ego_location):
        self.world = world
        self.patch_size = patch_size
        self.projectors_array = self.get_projectors_array(ego_location)

    def get_projectors_array(self, ego_location):
        bp_lib = self.world.get_blueprint_library()
        spawn_point = ego_location

        def get_combs(ele_list: list, comb_len=3) -> List[List[int or float]]:
            if comb_len > 1:
                return [[ele] + comb for ele in ele_list for comb in get_combs(ele_list, comb_len - 1)]
            else:
                return [[ele] for ele in ele_list]

        rotations = [-10, 10, 20]
        locations = [5, -5]
        fovs = [45]

        projectors = []
        for fov in fovs:
            for rotation in get_combs(rotations, 2):
                for location in get_combs(locations, 3):
                    # no roll
                    rotation = rotation + [0]
                    prj_depth_camera_bp = bp_lib.find('sensor.camera.depth')
                    prj_depth_camera_bp.set_attribute('fov', str(fov))
                    prj_depth_camera_bp.set_attribute('image_size_x', str(self.patch_size[0]))
                    prj_depth_camera_bp.set_attribute('image_size_y', str(self.patch_size[1]))

                    ve_x, ve_y, ve_z = spawn_point.location.x, spawn_point.location.y, spawn_point.location.z
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

    def __call__(self, ):
        objects_p3d = []
        objects_color = []
        for i, camera in enumerate(self.projectors_array):
            prj_depth_image_queue = camera['data_queue']
            prj_depth_camera = camera['projector']

            prj_depth_image = prj_depth_image_queue.get()
            carla_patch = CarlaPatch2Prj('../patch.npy')
            object_p3d, object_color = carla_patch.data2pcd(prj_depth_image, prj_depth_camera)
            objects_p3d.append(object_p3d)
            objects_color.append(object_color)
            print('project patch to world...')
            prj_depth_camera.destroy()
            print(f'destroy projector{i}')

        patch_selector = PatchSelector(objects_p3d, objects_color)
        return patch_selector()