import os.path

from collada import Collada
import collada
from OpenGL.GL import *
from OpenGL.GLUT import *

from scipy.spatial.transform import Rotation
from ctypes import c_float, c_void_p, sizeof
import numpy as np
from collada.polylist import Polylist
from collada.scene import ControllerNode
from ipywidgets import Controller
from scipy.constants import degree, value
from smplx.lbs import batch_rodrigues

from .transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp, rotation_matrix


class Joint:
    def __init__(self, id, inverse_transform_matrix):
        self.id = id
        self.children = []
        self.parent = None
        self.inverse_transform_matrix = inverse_transform_matrix


class KeyFrame:
    def __init__(self, time, joint_transform):
        self.time = time
        self.init_transform(joint_transform)

    def init_transform(self, joint_transform):
        self.joint_transform = dict()
        for key, value in joint_transform.items():
            translation_matrix = np.identity(4)
            translation_matrix[0, 3], translation_matrix[1, 3], translation_matrix[2, 3] = value[0, 3], value[1, 3], \
                                                                                           value[2, 3]

            rotation_matrix = quaternion_from_matrix(value)

            self.joint_transform[key] = [translation_matrix, rotation_matrix]


class ColladaModel:
    def __init__(self, collada_file_path):
        model = Collada(collada_file_path)
        self.vao = []
        self.ntriangles = []
        self.texture = []
        if not collada_file_path.endswith("human.dae"):
            self.inverse_transform_matrices = []
            self.joints_order = {}
            self.joints_matrix_map = {}
            for controller in model.controllers:
                controller_joint_matrices = list(controller.joint_matrices.values())
                for index, joint_name in enumerate(np.atleast_1d(np.squeeze(controller.weight_joints.data))):
                    if self.joints_matrix_map.get(joint_name) is None:
                        self.joints_matrix_map[joint_name] = controller_joint_matrices[index]
                    elif np.not_equal(self.joints_matrix_map[joint_name], controller_joint_matrices[index]).any():
                        print("Oh, no!!!!!")
                    if collada_file_path.endswith("Ramy.dae"):
                        if self.joints_order.get("Armature_" + joint_name) is None:
                            self.inverse_transform_matrices.append(controller_joint_matrices[index])
                            self.joints_order["Armature_" + joint_name] = len(self.inverse_transform_matrices)-1
                    else:
                        if self.joints_order.get(joint_name) is None:
                            self.inverse_transform_matrices.append(controller_joint_matrices[index])
                            self.joints_order[joint_name] = len(self.inverse_transform_matrices)-1
            # self.inverse_transform_matrices = [value for _, value in model.controllers[0].joint_matrices.items()]
            # self.inverse_transform_matrices = np.array(self.inverse_transform_matrices)
            # self.joints_order = {"Armature_" + joint_name: index for index, joint_name in
            #                      enumerate(np.squeeze(model.controllers[0].weight_joints.data))}
        else:
            self.inverse_transform_matrices = [value for _, value in model.controllers[0].joint_matrices.items()]
            self.joints_order = {"Armature_" + joint_name: index for index, joint_name in
                                 enumerate(np.squeeze(model.controllers[0].weight_joints.data))}
            self.joints_matrix_map = {}
            for controller in model.controllers:
                controller_joint_matrices = list(controller.joint_matrices.values())
                for index, joint_name in enumerate(np.atleast_1d(np.squeeze(controller.weight_joints.data))):
                    if self.joints_matrix_map.get(joint_name) is None:
                        self.joints_matrix_map[joint_name] = controller_joint_matrices[index]
                    elif np.not_equal(self.joints_matrix_map[joint_name], controller_joint_matrices[index]).any():
                        print("Oh, no!!!!!")

        self.joint_count = len(self.joints_order)
        # unbind_bone_name = ["mixamorig_LeftHandThumb4", "mixamorig_LeftHandIndex4", "mixamorig_LeftHandMiddle4", "mixamorig_LeftHandRing4", "mixamorig_LeftHandPinky4",
        #                          "mixamorig_RightHandThumb4", "mixamorig_RightHandIndex4", "mixamorig_RightHandMiddle4", "mixamorig_RightHandRing4", "mixamorig_RightHandPinky4"]
        # self.unbind_bone_index = np.array([self.joints_order[joint_name] for joint_name in unbind_bone_name])

        for node in model.scenes[0].nodes:
            if collada_file_path.endswith("Reaction.dae") or collada_file_path.endswith("nonPBR.dae"):
                if node.id == "mixamorig_Hips" or node.id == "mixamorig8_Hips":
                    self.root_joint = Joint(node.id,
                                            self.inverse_transform_matrices[self.joints_order.get(node.id)])
                    self.root_joint.children.extend(self.__load_armature(node))
                    self.rest_pose_joint_animation_matrix = {}
                    self.__calculate_rest_pose_animation_matrix(self.root_joint, np.identity(4))
                    self.joint_dict = {}
                    self.__construct_joint_dict(self.root_joint, None)

                for child in node.children:
                    if isinstance(child, ControllerNode):
                        self.__load_mesh_data(child)
            if node.id == 'Armature':
                self.root_joint = Joint(node.children[0].id.replace("Armature_", "", 1),
                                        self.inverse_transform_matrices[self.joints_order.get(node.children[0].id.replace("Armature_", "", 1))])
                self.root_joint.children.extend(self.__load_armature(node.children[0]))
                del self.inverse_transform_matrices
                self.rest_pose_joint_animation_matrix = {}
                self.__calculate_rest_pose_animation_matrix(self.root_joint, np.identity(4))
                self.joint_dict = {}
                self.__construct_joint_dict(self.root_joint, None)
                if not collada_file_path.endswith("human.dae"):
                    for i in range(1, len(node.children)):
                        self.__load_mesh_data(node.children[i].children[0])
                    # self.__load_mesh_data(node.children[2].children[0])
                    # # self.__load_mesh_data(node.children[3].children[0])
                    # self.__load_mesh_data(node.children[4].children[0])
                    # self.__load_mesh_data(node.children[5].children[0])
                    # self.__load_mesh_data(node.children[6].children[0])
                    # self.__load_mesh_data(node.children[7].children[0])
                    # self.__load_mesh_data(node.children[8].children[0])

            if node.id == "Cube":
                self.__load_mesh_data(node.children[0])

        self.render_static_matrices = [np.identity(4) for _ in range(len(self.joints_order))]
        # self.render_animation_matrices = [i for i in range(len(self.joints_order))]

        import json
        with open("SMPL-X skeleton.json", "r") as f:
            smplx_index2joint = json.load(f)
            self.smplx_index2joint = {int(key): value for key, value in smplx_index2joint.items()}
            self.smplx_joint2index = {value: key for key, value in self.smplx_index2joint.items()}
        with open("smplx2mixamo.json", "r") as f:
        # with open("smplx2sk.json", "r") as f:
            mixamo2smplx_joints_map = json.load(f)
            self.smplx2mixamo_joints_map = {value: key for key, value in mixamo2smplx_joints_map.items()}
            self.mixamo2smplx_joints_map = mixamo2smplx_joints_map
        with open("smplx_finger_transform.json", "r") as f:
            self.smplx_finger_euler = json.load(f)
            key_lists = list(self.smplx_finger_euler.keys())
            self.smplx_finger_rot_dict = {}
            smpl_rot = np.linalg.inv(self.root_joint.inverse_transform_matrix[:3, :3])
            for i, key in enumerate(key_lists):
                rot_matrix = Rotation.from_euler("xyz", self.smplx_finger_euler[key], degrees=True).as_matrix()
                if i%3 == 0:
                    self.smplx_finger_rot_dict[key] = np.matmul(smpl_rot, rot_matrix)
                else:
                    parent_key = key_lists[i-1]
                    self.smplx_finger_rot_dict[key] = np.matmul(self.smplx_finger_rot_dict[parent_key], rot_matrix)
        self.keyframes = []

        # self.__load_keyframes(model.animations, collada_file_path)
        self.__load_keyframes_from_smplx("/home/hhm/pose_process/data_vis/2")

        self.doing_animation = False
        self.frame_start_time = None
        self.animation_keyframe_pointer = 0

        # self.__set_joint_angle(self.root_joint.children[0], [1.57, 0.67, 0])
        # self.smplx_poses = np.array([-0.013232104480266571, -0.2054843306541443, -0.05122341960668564, -0.19700564444065094, -0.025097915902733803, 0.06495499610900879, 0.11946099996566772, -0.03601017966866493, 0.10146261006593704, -0.000832775083836168, 0.09232071042060852, -0.026628149673342705, 0.34644532203674316, 0.02861165814101696, -0.018566878512501717, -0.159354105591774, 0.04026065766811371, -0.07092364877462387, -0.05641816556453705, 0.014636288397014141, 0.020786477252840996, -0.06835556030273438, 0.25145798921585083, 0.009306746535003185, 0.10040702670812607, 0.010644057765603065, 0.14624226093292236, 0.06015906482934952, 0.07845903933048248, 0.07417913526296616, -0.025629688054323196, 0.030130499973893166, -0.025519341230392456, 0.05201706290245056, 0.005465198308229446, 0.05324413254857063, 0.25757601857185364, 0.5102375745773315, -0.1355082392692566, 0.025052731856703758, 0.1398579627275467, 0.004772544372826815, 0.14772482216358185, -0.05554765835404396, 0.27120494842529297, -0.04341213032603264, 0.5872477293014526, -0.03574392944574356, 0.36285415291786194, -0.2700650990009308, -1.0233166217803955, 0.25014087557792664, 0.20533864200115204, 0.8836807012557983, 0.18478870391845703, -1.7285561561584473, -0.15410436689853668, 0.18165907263755798, 0.31051573157310486, 0.09037993848323822, -0.662392795085907, -0.24611853063106537, 0.42543184757232666, -0.14846445620059967, 0.16134457290172577, -0.09032043814659119, 0.048781100660562515, 0.0020039950031787157, 0.0038107242435216904, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01805310882627964, 0.18681661784648895, -1.067522406578064, 0.38887760043144226, -0.07576794922351837, -1.2938296794891357, -0.2534017860889435, -0.22823816537857056, 0.062493160367012024, -0.4997025728225708, 0.055100638419389725, -1.6000711917877197, -0.08493291586637497, 0.11063466221094131, -0.9430750608444214, -0.011919503100216389, -0.12384437024593353, -0.5485368371009827, -0.5719927549362183, -0.7005698680877686, -0.9286771416664124, -0.9715783596038818, -0.1196148470044136, -0.9070805907249451, -0.3850429654121399, -0.2095586657524109, -0.705706000328064, -0.266874760389328, -0.41972097754478455, -1.5415629148483276, -0.6304079294204712, -0.2759303152561188, -1.0951869487762451, -0.29364481568336487, -0.02338808961212635, -0.6974433064460754, 0.9681151509284973, 0.500689685344696, -0.03594258800148964, -0.6188196539878845, -0.2210063934326172, 0.08410488069057465, 0.570131778717041, -0.11006329208612442, -0.16703303158283234, 0.125240758061409, -0.06388267129659653, 0.733893871307373, 0.2574699819087982, 0.12979720532894135, 0.908207356929779, -0.2708408534526825, 0.12398278713226318, -0.1524866670370102, -0.28568530082702637, 0.009124931879341602, 0.849337637424469, -0.11575823277235031, -0.12757940590381622, 0.7795430421829224, 0.028035325929522514, 0.1195296049118042, 0.18480762839317322, -0.3899335265159607, 0.37081822752952576, 0.43408724665641785, -0.4098811149597168, -0.025760484859347343, 0.5803067684173584, -0.26975882053375244, 0.06586438417434692, 0.15986722707748413, -0.17501528561115265, 0.22303248941898346, 0.8296855092048645, -0.3454570174217224, 0.07300589233636856, 0.6966943740844727, -0.1755705624818802, 0.10920323431491852, 0.2908267676830292, 1.2211940288543701, -0.6028814315795898, 0.3054938316345215, -0.5468103885650635, 0.04367611184716225, -0.4370841979980469, 0.5432970523834229, 0.24968038499355316, 0.4824322760105133])
        # self.smplx_poses = self.smplx_poses.reshape(-1, 3)


        # self.__load_pose_from_pkl("/home/hhm/smpl_model/smplx_poses/smplx_gt/trainset_3dpeople_adults_bfh/10004_w_Amaya_0_0.pkl")
        # self.__set_joint_angle("mixamorig_LeftForeArm", [30, 0, 0])
        # self.__set_joint_angle("mixamorig_LeftArm", [45, 0, 0])
        # self.rest_pose_joint_animation_matrix["mixamorig_LeftShoulder"] = self.__set_joint_angle("mixamorig_LeftShoulder", self.smplx_poses[13])
        # self.rest_pose_joint_animation_matrix["mixamorig_LeftArm"] = self.__set_joint_angle("mixamorig_LeftArm",
        #                                                                   self.smplx_poses[16])
        #self.__set_joint_angle(self.joint_dict["mixamorig_LeftForeArm"], self.smplx_body_poses[0, 17])
        # self.__set_joint_angle(self.root_joint.children[1].children[0], self.rest_pose_joint_animation_matrix[self.root_joint.id], [-45, 0, 0])
        # self.__set_joint_angle(self.root_joint.children[2], [1.57, 0.67, 0])

    def __load_pose_from_pkl(self, file_name):
        import pickle
        with open(file_name, 'rb') as f:
            poses_dict= pickle.load(f)
        pose_data = np.concatenate((poses_dict["global_orient"][0], poses_dict["body_pose"][0], poses_dict["jaw_pose"][0], poses_dict["leye_pose"][0], poses_dict["reye_pose"][0], poses_dict["left_hand_pose"][0], poses_dict["right_hand_pose"][0]))
        pose_data = pose_data.reshape(-1, 3)
        for index in range(len(pose_data)):
            pose = pose_data[index]
            smplx_joint_name = self.smplx_index2joint[index]
            if self.smplx2mixamo_joints_map.get(smplx_joint_name) is not None:
                mixamo_joint_name = self.smplx2mixamo_joints_map.get(smplx_joint_name)
                angle = Rotation.from_rotvec(pose).as_matrix()
                self.__set_joint_angle(mixamo_joint_name, angle)
                # self.rest_pose_joint_animation_matrix[mixamo_joint_name] = self.__extract_joint_angle_from_smplx_pose(mixamo_joint_name, pose)

    def __set_joint_angle(self, mixamo_joint_name, angle):
        from scipy.spatial.transform import Rotation
        rot_matrix_ = Rotation.from_euler("xyz", angle, degrees=True).as_matrix()
        M_joint2world = np.linalg.inv(self.joints_matrix_map[mixamo_joint_name])
        M_rotation_joint2world = np.identity(4)
        M_rotation_joint2world[:3, :3] = M_joint2world[:3, :3]
        M_translation_joint2world = np.identity(4)
        M_translation_joint2world[:3, 3] = M_joint2world[:3, 3]
        joint_animation_matrix = self.rest_pose_joint_animation_matrix[mixamo_joint_name]
        joint_animation_matrix[:3,:3] = np.matmul(self.rest_pose_joint_animation_matrix[mixamo_joint_name][:3,:3], rot_matrix_)
        self.rest_pose_joint_animation_matrix[mixamo_joint_name] = joint_animation_matrix

    def __extract_joint_angle_from_smplx_pose(self, mixamo_joint_name, angle):
        matrix_ = Rotation.from_rotvec(angle).as_matrix()
        smplx_joint_name = self.mixamo2smplx_joints_map[mixamo_joint_name]
        smplx_index = self.smplx_joint2index[smplx_joint_name]

        # calculate transformation matrix
        smpl_rot = np.linalg.inv(self.root_joint.inverse_transform_matrix[:3,:3])
        smplx_initial_rot_matrix = np.identity(3)
        if smplx_joint_name in self.smplx_finger_rot_dict:
            # smpl_rot = self.smplx_finger_rot_dict[smplx_joint_name]
            smplx_euler = self.smplx_finger_euler[smplx_joint_name]
            smplx_initial_rot_matrix = Rotation.from_euler("xyz", smplx_euler, degrees=True).as_matrix()

        mixamo_rot = np.linalg.inv(self.joints_matrix_map[mixamo_joint_name][:3,:3])
        smpl2mixamo_rot = np.matmul(np.linalg.inv(mixamo_rot), smpl_rot)
        matrix_ = np.matmul(smplx_initial_rot_matrix, matrix_)
        matrix_ = np.matmul(np.matmul(smpl2mixamo_rot, matrix_), np.linalg.inv(smpl2mixamo_rot))
        # rot_matrix = Rotation.from_euler('xyz', euler, degrees=True).as_matrix()
        trans_matrix = np.identity(4)
        # trans_matrix[:3, :3] = np.linalg.inv(rot_matrix)
        trans_matrix[:3, :3] = matrix_
        joint_animation_matrix = np.matmul(self.rest_pose_joint_animation_matrix[mixamo_joint_name], trans_matrix)
        return joint_animation_matrix
        # self.rest_pose_joint_animation_matrix[joint.id] = np.identity(4)

    def __construct_joint_dict(self, joint:Joint, joint_parent:Joint):
        self.joint_dict[joint.id.replace("Armature_", "", 1)] = joint
        joint.parent = joint_parent
        for child in joint.children:
            self.__construct_joint_dict(child, joint)

    def __calculate_rest_pose_animation_matrix(self, joint:Joint, parent_matrix):
        animation_matrix =  np.matmul(np.linalg.inv(parent_matrix), np.linalg.inv(self.joints_matrix_map[joint.id.replace("Armature_", "", 1)]))
        self.rest_pose_joint_animation_matrix[joint.id.replace("Armature_", "", 1)] = animation_matrix
        parent_matrix = np.matmul(parent_matrix, animation_matrix)
        for child in joint.children:
            self.__calculate_rest_pose_animation_matrix(child, parent_matrix)

    def __load_keyframes_from_smplx(self, directory):
        import re
        import smplx
        import pickle
        import torch
        file_list = os.listdir(directory)
        # file_list = sorted(file_list, key=lambda f: int(re.search(r"toy_(\d+)\.pkl", f).group(1)))
        file_list = sorted(file_list, key=lambda f: int(re.search(r"\S+_(\d+)_3D\.pkl$", f).group(1)))
        frame_number = len(file_list)

        # cpnstant number
        frame_rate = 1/15
        for i, file_name in enumerate(file_list):
            time = frame_rate * i
            joint_dict = {}
            import pickle
            with open(os.path.join(directory, file_name), 'rb') as f:
                poses_dict = pickle.load(f)
            # pose_data = np.concatenate((poses_dict["global_orient"], poses_dict["body_pose"],
            #                             poses_dict["jaw_pose"], [0, 0, 0],
            #                             [0, 0, 0], poses_dict["left_hand_pose"],
            #                             poses_dict["right_hand_pose"]))
            pose_data = np.concatenate(([0, 0, 0], poses_dict["smplx_body_pose"],
                                        poses_dict["smplx_jaw_pose"], [0, 0, 0],
                                        [0, 0, 0], poses_dict["smplx_lhand_pose"],
                                        poses_dict["smplx_rhand_pose"]))
            pose_data = pose_data.reshape(-1, 3)
            for index in range(len(pose_data)):
                pose = pose_data[index]
                smplx_joint_name = self.smplx_index2joint[index]
                # if smplx_joint_name in ["left_index2", "right_index2", "left_middle2", "right_middle2", "left_pinky2", "right_pinky2", "left_ring2", "right_ring2", "left_thumb2", "right_thumb2"]:
                #     mixamo_joint_name = self.smplx2mixamo_joints_map.get(smplx_joint_name)
                #     mixamo_joint_name1 = mixamo_joint_name[:-1] + "2"
                #     mixamo_joint_name2 = mixamo_joint_name[:-1] + "3"
                #     joint_dict[mixamo_joint_name1] = self.__set_joint_angle(mixamo_joint_name1, pose * 1 /3)
                #     joint_dict[mixamo_joint_name2] = self.__set_joint_angle(mixamo_joint_name2, pose * 1 /3)
                # elif self.smplx2mixamo_joints_map.get(smplx_joint_name) is not None:
                if self.smplx2mixamo_joints_map.get(smplx_joint_name) is not None:
                    mixamo_joint_name = self.smplx2mixamo_joints_map.get(smplx_joint_name)
                    joint_dict[mixamo_joint_name] = self.__extract_joint_angle_from_smplx_pose(mixamo_joint_name, pose)
            self.keyframes.append(KeyFrame(time, joint_dict))

    def __load_keyframes(self, animation_node, collada_file_path):
        self.keyframes = []
        if collada_file_path.endswith("human.dae"):
            keyframes_times = np.squeeze(animation_node[0].sourceById.get(animation_node[0].id + "-input").data).tolist()
            for index, time in enumerate(keyframes_times):
                joint_dict = dict()
                for animation in animation_node:
                    joint_name = "_".join(animation.id.split("_")[1:-2])
                    joint_dict[animation.id] = animation.sourceById.get(animation.id + "-output").data[
                                               index * 16:(index + 1) * 16].reshape((4, 4))
                self.keyframes.append(KeyFrame(time, joint_dict))
        else:
            node = animation_node[0]
            id = node.id.split("-")[0]
            keyframes_times = np.squeeze(node.sourceById.get(id + "-Matrix-animation-input").data).tolist()
            for index, time in enumerate(keyframes_times):
                joint_dict = dict()
                for animation in animation_node:
                    id = animation.id.split("-")[0]
                    data = animation.sourceById.get(id + "-Matrix-animation-output-transform").data
                    joint_dict[id] = data[index * 16:(index + 1) * 16].reshape((4, 4))
                    # joint_dict[id][:3, 3] = np.zeros((1, 3))
                self.keyframes.append(KeyFrame(time, joint_dict))
            # time = 0.0
            # joint_dict = dict()
            # for joint_name in self.joints_matrix_map.keys():
            #     joint_key = joint_name + "_pose_matrix"
            #     joint_dict[joint_key] = np.identity(4)
            # self.keyframes.append(KeyFrame(time, joint_dict))
            # self.keyframes.append(KeyFrame(time+0.2, joint_dict))
            # self.keyframes.append(KeyFrame(time +0.4, joint_dict))


    def __load_armature(self, node):
        children = []
        for child in node.children:
            if type(child) == collada.scene.Node:
                if self.joints_order.get(child.id.replace("Armature_", "", 1))!=None:
                    joint = Joint(child.id, self.inverse_transform_matrices[self.joints_order.get(child.id.replace("Armature_", "", 1))])
                    joint.children.extend(self.__load_armature(child))
                    children.append(joint)
                else:
                    print("What the hell")
        return children

    def __load_mesh_data(self, node):
        weights_data = np.squeeze(node.controller.weights.data)
        weights_joint = np.squeeze(node.controller.weight_joints.data, axis=1)
        for index, mesh_data in enumerate(node.controller.geometry.primitives):
            vertex = []
            self.ntriangles.append(len(mesh_data))
            try:
                material = node.materials[index]
                diffuse = material.target.effect.diffuse
                texture_type = "v_color" if type(diffuse) == tuple else "sampler"
            except:
                texture_type = None
            for i in range(len(mesh_data)):
                if isinstance(mesh_data, Polylist):
                    v = mesh_data.vertex[mesh_data.vertex_index[mesh_data.polystarts[i]:mesh_data.polyends[i]]]
                    n = mesh_data.normal[mesh_data.normal_index[mesh_data.polystarts[i]:mesh_data.polyends[i]]]
                    if texture_type == "sampler":
                        t = mesh_data.texcoordset[0][mesh_data.texcoord_indexset[0][mesh_data.polystarts[i]:mesh_data.polyends[i]]]
                    elif texture_type == "v_color":
                        t = np.array(diffuse[:-1]).reshape([1, -1]).repeat([3], axis=0)
                    j_index_ = []
                    w_index = []
                    for vertex_index in list(mesh_data.vertex_index[mesh_data.polystarts[i]:mesh_data.polyends[i]]):
                        joint_name_array = list(weights_joint[node.controller.joint_index[vertex_index]])
                        joint_index = np.array([self.joints_order[joint_name] for joint_name in joint_name_array])
                        # joint_index = joint_index[~np.isin(joint_index, self.unbind_bone_index)]
                        old_joint_index = node.controller.joint_index[vertex_index]
                        j_index_.append(joint_index)
                        w_index.append(node.controller.weight_index[vertex_index])

                    w_ = [weights_data[index] for index in w_index]
                else:
                    v = mesh_data.vertex[mesh_data.vertex_index[i]]
                    n = mesh_data.normal[mesh_data.normal_index[i]]
                    if texture_type == "sampler":
                        t = mesh_data.texcoordset[0][mesh_data.texcoord_indexset[0][i]]
                    elif texture_type == "v_color":
                        t = np.array(diffuse[:-1]).reshape([1, -1]).repeat([3], axis=0)
                    joint_name_array = [weights_joint[node.controller.joint_index[mesh_data.vertex_index[i,j]]] for j in range(3)]
                    j_index_ = []
                    for j in range(3):
                        j_index_.append(np.array([self.joints_order[joint_name] for joint_name in joint_name_array[j]]))
                    # j_index_ = [node.controller.joint_index[mesh_data.vertex_index[i, 0]],
                    #             node.controller.joint_index[mesh_data.vertex_index[i, 1]],
                    #             node.controller.joint_index[mesh_data.vertex_index[i, 2]]]

                    w_index = [node.controller.weight_index[mesh_data.vertex_index[i, 0]],
                               node.controller.weight_index[mesh_data.vertex_index[i, 1]],
                               node.controller.weight_index[mesh_data.vertex_index[i, 2]]]

                    w_ = [weights_data[w_index[0]], weights_data[w_index[1]], weights_data[w_index[2]]]

                j_index = []
                w = []
                for j in range(len(j_index_)):
                    if j_index_[j].size < 9:
                        j_index.append(
                            np.pad(j_index_[j], (0, 9 - j_index_[j].size), 'constant', constant_values=(0, 0))[:9])
                        w.append(
                            np.pad(w_[j]/np.sum(w_[j][:9]), (0, 9 - j_index_[j].size), 'constant', constant_values=(0, 0))[:9])
                    else:
                        j_index.append(j_index_[j][:9])

                        w.append(w_[j][:9] / np.sum(w_[j][:9]))

                if not texture_type:
                    vertex.append(np.concatenate((v, n, j_index, w), axis=1))
                else:
                    vertex.append(np.concatenate((v, n, j_index, w, t), axis=1))

            self.__set_vao(np.row_stack(vertex), texture_type)

            if texture_type == "sampler":
                self.texture.append(self.__set_texture(diffuse.sampler.surface.image))
            else:
                self.texture.append(-1)

    def __set_vao(self, points, texture_type):
        points = np.squeeze(points).astype(np.float32)
        pos = points[:, :3]
        joint_index = points[:, 6:9]
        self.vao.append(glGenVertexArrays(1))
        vbo = glGenBuffers(1)
        glBindVertexArray(self.vao[-1])

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, points, GL_STATIC_DRAW)

        step = 26 if texture_type == "sampler" else 27 if texture_type == "v_color" else 24

        glVertexAttribPointer(0, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(0 * sizeof(c_float)))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(3 * sizeof(c_float)))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(6 * sizeof(c_float)))
        glVertexAttribPointer(3, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(9 * sizeof(c_float)))
        glVertexAttribPointer(4, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(12 * sizeof(c_float)))
        glEnableVertexAttribArray(2)
        glEnableVertexAttribArray(3)
        glEnableVertexAttribArray(4)

        glVertexAttribPointer(5, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(15 * sizeof(c_float)))
        glVertexAttribPointer(6, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(18 * sizeof(c_float)))
        glVertexAttribPointer(7, 3, GL_FLOAT, False, step * sizeof(c_float), c_void_p(21 * sizeof(c_float)))
        glEnableVertexAttribArray(5)
        glEnableVertexAttribArray(6)
        glEnableVertexAttribArray(7)

        if texture_type:
            glVertexAttribPointer(8, 2 if texture_type == "sampler" else 3, GL_FLOAT, False, step * sizeof(c_float),
                                  c_void_p(24 * sizeof(c_float)))

        glEnableVertexAttribArray(8)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def __set_texture(self, image):
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        image = image.pilimage
        try:
            ix, iy, image = image.size[0], image.size[1], image.tobytes("raw", "RGBA", 0, -1)
        except:
            ix, iy, image = image.size[0], image.size[1], image.tobytes("raw", "RGBX", 0, -1)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glGenerateMipmap(GL_TEXTURE_2D)
        return texture

    def render(self, shader_program):
        shader_program.use()

        for index, m in enumerate(self.render_static_matrices):
            shader_program.set_matrix("jointTransforms[" + str(index) + "]", m, transpose=GL_TRUE)

        for index, vao in enumerate(self.vao):
            if self.texture[index] != -1:
                glUniform1i(glGetUniformLocation(shader_program.id, "texture1"), index)
                glActiveTexture(GL_TEXTURE0 + index)
                glBindTexture(GL_TEXTURE_2D, self.texture[index])

            glBindVertexArray(vao)

            glDrawArrays(GL_TRIANGLES, 0, self.ntriangles[index]*3)
            if self.texture[index] != -1:
                glBindTexture(GL_TEXTURE_2D, 0)

    def animation(self, shader_program, loop_animation=False):
        if not self.doing_animation:
            self.doing_animation = True
            self.frame_start_time = glutGet(GLUT_ELAPSED_TIME)
        self.interpolation_joint = dict()
        if len(self.keyframes)>0:
            pre_frame, next_frame = self.keyframes[self.animation_keyframe_pointer:self.animation_keyframe_pointer + 2]
            frame_duration_time = (next_frame.time - pre_frame.time) * 1000
            current_frame_time = glutGet(GLUT_ELAPSED_TIME)
            frame_progress = (current_frame_time - self.frame_start_time) / frame_duration_time
            if frame_progress >= 1:
                self.animation_keyframe_pointer += 1
                if self.animation_keyframe_pointer == len(self.keyframes) - 1:
                    self.animation_keyframe_pointer = 0
                self.frame_start_time = glutGet(GLUT_ELAPSED_TIME)
                pre_frame, next_frame = self.keyframes[self.animation_keyframe_pointer:self.animation_keyframe_pointer + 2]
                frame_duration_time = (next_frame.time - pre_frame.time) * 1000
                current_frame_time = glutGet(GLUT_ELAPSED_TIME)
                frame_progress = (current_frame_time - self.frame_start_time) / frame_duration_time

            # interpolating; pre_frame, next_frame, frame_progress
            # from scipy.spatial.transform import Rotation
            # for key, value in self.keyframes[100].joint_transform.items():
            #     i_translation = np.identity(4)
            #     i_translation[:3,:3] = Rotation.from_quat(value[1][[1, 2, 3, 0]]).as_matrix()
            #     self.interpolation_joint[key] = np.matmul(value[0], i_translation)
            for key, value in pre_frame.joint_transform.items():
                t_m = self.interpolating_translation(value[0], next_frame.joint_transform.get(key)[0], frame_progress)
                r_m = self.interpolating_rotation(value[1], next_frame.joint_transform.get(key)[1], frame_progress)
                matrix = np.matmul(t_m, r_m)
                self.interpolation_joint[key] = matrix

        self.load_animation_matrices(self.root_joint, np.identity(4))
        self.render(shader_program)

    def interpolating_translation(self, translation_a, translation_b, progress):
        i_translation = np.identity(4)
        i_translation[0, 3] = translation_a[0, 3] + (translation_b[0, 3] - translation_a[0, 3]) * progress
        i_translation[1, 3] = translation_a[1, 3] + (translation_b[1, 3] - translation_a[1, 3]) * progress
        i_translation[2, 3] = translation_a[2, 3] + (translation_b[2, 3] - translation_a[2, 3]) * progress
        return i_translation

    def interpolating_rotation(self, rotation_a, rotation_b, progress):
        return quaternion_matrix(quaternion_slerp(rotation_a, rotation_b, progress))

    def load_animation_matrices(self, joint, parent_matrix):
        if self.interpolation_joint.get(joint.id + "_pose_matrix") is None:
            if self.interpolation_joint.get(joint.id.replace("Armature_", "", 1)) is None:
                p = np.matmul(parent_matrix, self.rest_pose_joint_animation_matrix[joint.id.replace("Armature_", "", 1)])
                # p = np.matmul(parent_matrix, np.identity(4))
            else:
                p = np.matmul(parent_matrix, self.interpolation_joint.get(joint.id.replace("Armature_", "", 1)))
        else:
            p = np.matmul(parent_matrix, self.interpolation_joint.get(joint.id + "_pose_matrix"))
        for child in joint.children:
            self.load_animation_matrices(child, p)
        matrix_ret = np.matmul(p, joint.inverse_transform_matrix)
        # if joint.id == "mixamorig_LeftLeg":
        #     from scipy.spatial.transform import Rotation
        #     rot_matrix = Rotation.from_euler('xyz', [40, 0, 0], degrees=True).as_matrix()
        #     trans_matrix = np.identity(4)
        #     trans_matrix[:3, :3] = rot_matrix
        #     matrix_ret = np.matmul(trans_matrix, matrix_ret)
        self.render_static_matrices[self.joints_order.get(joint.id.replace("Armature_", "", 1))] = matrix_ret


if __name__ == "__main__":
    # scene = ColladaModel("/home/shuai/human.dae")

    # a = np.array([[3], [14], [15], [12], [111], [134]])
    #
    # a_sque = np.squeeze(a)

    pass
