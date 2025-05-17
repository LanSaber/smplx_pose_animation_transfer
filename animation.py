import os.path

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np

from shader import ShaderProgram
from model import Model, ModelFromExport
import glm
import time
from camera3D import Camera3D
from keyboard import keys_down, keys_up
from collada_loader.model_loader import ColladaModel
import argparse

parser = argparse.ArgumentParser()

# Define expected arguments
parser.add_argument("--sequence_index", type=int, help="sequence index", default=0)

args = parser.parse_args()


SCR_WIDTH = 800
SCR_HEIGHT = 800

ego_centric = False

# camera = Camera3D(glm.vec3(0.0, 0.0, 100.0))
# camera = Camera3D(glm.vec3(0.0, 100.0, -100.0))
# camera = Camera3D(glm.vec3(0.0, 1.5, 1.5))
if not ego_centric:
    camera = Camera3D(glm.vec3(0.0, 130, 280))
    projection = glm.perspective(glm.radians(30.0), SCR_WIDTH * 1.0 / SCR_HEIGHT, 0.1, 1000)
    # Lighting and camera position
    light_pos = np.array([0, 100.0, 200.0], dtype=np.float32)
else:
    camera = Camera3D(glm.vec3(0.0, 160, 20), front=(0.0, 0.0, 1.0), yaw=90, pitch=-45)
    projection = glm.perspective(glm.radians(120.0), SCR_WIDTH * 1.0 / SCR_HEIGHT, 0.1, 1000)
    # projection = glm.perspective(glm.radians(60.0), SCR_WIDTH * 1.0 / SCR_HEIGHT, 0.1, 1000)
    # Lighting and camera position
    light_pos = np.array([0, 300.0, 0.0], dtype=np.float32)

last_x = SCR_WIDTH / 2.0
last_y = SCR_HEIGHT / 2.0
first_mouse = True
mouse_leave = True

delta_time = 0.0
last_frame = 0.0

fps_count = 0
_fps = 0

NUM_SAMPLES = 10
frameTimes = []
currentFrame = 0
prevTicks = 0
_frameTime = None

shader_program = None  # type:ShaderProgram
robot_program = None  # type:ShaderProgram
grid_model = None  # type:Model
human_model = None  # type:ColladaModel

grid_position = [
    glm.vec3(1.0, 1.0, 1.0),
    [glm.radians(90), glm.vec3(1.0, 0.0, 0.0)],
    glm.vec3(0, 90.0, -10)
]

# Quad vertices (full-screen)
background_vertices = np.array([
    -1, -1, 0.0,  0.0, 0.0,  # Bottom left
     1, -1, 0.0,  1.0, 0.0,  # Bottom right
     1,  1, 0.0,  1.0, 1.0,  # Top right
    -1,  1, 0.0,  0.0, 1.0   # Top left
], dtype=np.float32)

indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)


def generate_grid_mesh(min, max, step=1.0):
    shape = int((max - min) // step)
    vertices = []
    indices = []
    r = np.arange(min, max, step)
    for i in range(shape):
        row_max = (i + 1) * shape - 1  # 行最大元素
        for j in range(shape):
            column_max = (shape - 1) * shape + j  # 列最大元素
            c_index = i * shape + j  # 当前索引位置
            c_right = c_index + 1  # 当前索引位置右边一个位置
            c_down = c_index + shape  # 当前索引下面一个位置
            if c_right <= row_max:  # 如果索引超过最右边
                indices.extend([c_index, c_right])
            if c_down <= column_max:  # 如果索引超过最下边
                indices.extend([c_index, c_down])

            vertices.extend([r[i], r[j], 0, 0, 0])

    return np.array(vertices, dtype=np.float32), indices

def generate_floor_mesh(min, max):
    vertices = [min, min, 0, 0, 0, min, max, 0, 1, 0, max, max, 0, 1, 1, max, min, 0, 0, 1]
    indices = [0, 1, 2, 2, 3, 0]
    return np.array(vertices, dtype=np.float32), indices



def init():
    grid_vertices, grid_mesh = generate_grid_mesh(-50, 60, step=10)
    floor_vertices, floor_mesh = generate_floor_mesh(-1, 1)

    global shader_program
    shader_program = ShaderProgram("resources/shaders/shader.vert", "resources/shaders/shader.frag")
    shader_program.init()

    global robot_program
    robot_program = ShaderProgram("resources/shaders/shader_robot.vert", "resources/shaders/shader_robot.frag")
    robot_program.init()

    global grid_model
    grid_model = Model([grid_vertices], indices=[grid_mesh], vertex_format="VT")

    global floor_model
    floor_model = Model([floor_vertices], indices=[floor_mesh], vertex_format="VT", texture_path=["background.jpg"])

    global human_model

    human_model = ColladaModel("resources/Ch07_nonPBR/Ch07_nonPBR.dae", args.sequence_index)
    # human_model = ColladaModel("resources/woman/Humano_Rig_052-6525_01_T-LOD0.dae", args.sequence_index)
    # human_model = ColladaModel("resources/Louise/louise1.dae")
    # human_model = ColladaModel("resources/ramy_changed/ramy.dae")
    # human_model = ColladaModel("resources/Reaction/Reaction.dae")
    # human_model = ColladaModel("resources/human.dae")
    glEnable(GL_DEPTH_TEST)


def drawFunc():
    # glClearColor(173.0/255, 216.0/255, 230.0/255, 0.0)
    glClearDepth(1.0)
    glPointSize(5)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    current_frame = glutGet(GLUT_ELAPSED_TIME)

    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    #projection = glm.ortho(-10.0, 10.0, -10.0, 10.0, 0.1, 200.0)
    view = camera.get_view_matrix()

    # Extract the rotation matrix (upper-left 3x3)
    R = glm.mat3(view)

    # Extract the translation vector (last column)
    T = glm.vec3(view[3])

    # Calculate the camera position in world space
    viewPos =  glm.vec3(-glm.transpose(R) * T)


    shader_program.use()
    shader_program.set_matrix("projection", glm.value_ptr(projection))
    shader_program.set_matrix("view", glm.value_ptr(view))

    m = glm.mat4(1.0)
    m = glm.translate(m, grid_position[2])
    m = glm.rotate(m, glm.radians(0), grid_position[1][1])
    m = glm.scale(m, glm.vec3(5))
    shader_program.set_matrix("model", glm.value_ptr(m))
    floor_model.draw(shader_program, draw_type=GL_TRIANGLES)
    # shader_program.un_use()
    # shader_program.use()
    # grid_color = glm.vec3(0.0, 0.2, 0.3)
    # glUniform3fv(glGetUniformLocation(shader_program.id, "fragColor"), 1, glm.value_ptr(grid_color))
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    # # grid_model.draw(shader_program, draw_type=GL_LINES)
    # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    shader_program.un_use()
    glEnable(GL_DEPTH_TEST)

    robot_program.use()
    robot_program.set_matrix("projection", glm.value_ptr(projection))
    # Pass view_position (vec3) to the shader instead of the entire matrix
    glUniform3fv(glGetUniformLocation(robot_program.id, "viewPos"), 1, glm.value_ptr(viewPos))


    glUniform3fv(glGetUniformLocation(robot_program.id, "lightPos"), 1, light_pos)
    glUniformMatrix4fv(glGetUniformLocation(robot_program.id, "view"), 1, GL_FALSE, glm.value_ptr(view))




    m = glm.mat4(1.0)
    # m = glm.rotate(m, glm.radians(-90), glm.vec3(1, 0, 0))
    robot_program.set_matrix("model", glm.value_ptr(m))

    # # Define the light properties
    # light_direction = np.array([0.0, -1.0, -1.0], dtype=np.float32)  # Direction of light
    # light_ambient = np.array([0.1, 0.1, 0.1], dtype=np.float32)  # Ambient intensity
    # light_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # Diffuse intensity
    # light_specular = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # Specular intensity
    #
    # # Set uniforms
    # glUniform3fv(glGetUniformLocation(shader_program, "lightDirection"), 1, light_direction)
    # glUniform3fv(glGetUniformLocation(shader_program, "lightAmbient"), 1, light_ambient)
    # glUniform3fv(glGetUniformLocation(shader_program, "lightDiffuse"), 1, light_diffuse)
    # glUniform3fv(glGetUniformLocation(shader_program, "lightSpecular"), 1, light_specular)

    robot_program.un_use()

    human_model.animation(robot_program)

    #
    global last_frame
    last_frame = glutGet(GLUT_ELAPSED_TIME)
    global delta_time
    delta_time = (last_frame - current_frame)
    camera.process_keyboard(delta_time / 1000)
    if delta_time < 16:
        time.sleep((16 - delta_time) / 1000)
    calculate_FPS()
    global fps_count
    if fps_count == 100:
        fps_count = 0
        # print('fps: %.2f' % _fps)
    fps_count += 1
    glutSwapBuffers()
    glutPostRedisplay()

    error = glGetError()
    if error != GL_NO_ERROR:
        print("OpenGL Error:", error)


def calculate_FPS():
    global prevTicks
    global _fps
    global currentFrame
    currentTicks = glutGet(GLUT_ELAPSED_TIME)
    _frameTime = currentTicks - prevTicks
    currentFrame += 1
    if currentFrame <= NUM_SAMPLES:
        frameTimes.append(_frameTime)
    else:
        frameTimes[(currentFrame) % NUM_SAMPLES] = _frameTime
    if currentFrame < NUM_SAMPLES:
        count_ = currentFrame
    else:
        count_ = NUM_SAMPLES

    frameTimeAverage = 0
    for i in range(count_):
        frameTimeAverage += frameTimes[i]

    prevTicks = currentTicks

    frameTimeAverage /= count_
    if frameTimeAverage > 0:
        _fps = 1000 / frameTimeAverage

    else:
        _fps = 0.00


def reshape(w, h):
    glViewport(0, 0, w, h)


def mouse_move(x, y):
    global last_x
    global last_y
    global first_mouse
    global mouse_leave

    if mouse_leave:
        last_x = x
        last_y = y
        mouse_leave = False

    if first_mouse:
        last_x = x
        last_y = y
        first_mouse = False

    x_offset = x - last_x

    y_offset = last_y - y

    last_x = x
    last_y = y

    # print(x, y)
    camera.process_mouse_movement(x_offset, y_offset)


def mouse_state(state):
    global mouse_leave
    glutWarpPointer(int(SCR_WIDTH / 2), int(SCR_HEIGHT / 2))
    if state == 1:
        mouse_leave = True
        pass


def main():
    glutInit()
    glutInitContextVersion(3, 3)
    glutInitContextProfile(GLUT_CORE_PROFILE)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT)
    glutCreateWindow(b"demo")
    # glutSetCursor(GLUT_CURSOR_NONE)
    # print(glGetString(GL_VERSION))

    global prevTicks
    prevTicks = glutGet(GLUT_ELAPSED_TIME)

    init()

    glutDisplayFunc(drawFunc)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keys_down)
    glutKeyboardUpFunc(keys_up)
    glutPassiveMotionFunc(mouse_move)
    glutMotionFunc(mouse_move)
    glutEntryFunc(mouse_state)

    glutMainLoop()


# Load image using pygame
def load_texture_img(image_path):
    from PIL import Image
    # global background_image
    background_image = Image.open(image_path)
    background_image = background_image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip because OpenGL's Y-axis is inverted
    # global background_image_data
    background_image_data = background_image.convert("RGB").tobytes()
    global background_texture
    background_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, background_texture)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, background_image.width, background_image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, background_image_data)

def play_pose_parameters(pose_dict, output_dir = "file"):
    human_model.keyframes.clear()
    human_model.load_keyframes_from_dict(pose_dict)
    image_list = []
    for i in range(len(human_model.keyframes)):
        __drawFunc(i, image_list)
    if not os.path.exists(os.path.join("render_result", output_dir)):
        os.mkdir(os.path.join("render_result", output_dir))
    for frame_idx, img in enumerate(image_list):
        img.save(os.path.join("render_result", output_dir, str(frame_idx) + ".png"))


def set_pose_data(pose_dict):
    human_model.keyframes.clear()
    human_model.load_keyframes_from_dict(pose_dict)

def get_image(i):
    image_list = []
    image = __drawFunc(i, image_list)
    image = image[:, :, [2, 1, 0]]
    return image



def save_pose_into_videos(pose_dict, output_dir="file", file_name="output_video"):
    human_model.keyframes.clear()
    human_model.load_keyframes_from_dict(pose_dict)
    image_list = []
    for i in range(len(human_model.keyframes)):
        __drawFunc(i, image_list)
    if not os.path.exists(os.path.join("render_result", output_dir)):
        os.mkdir(os.path.join("render_result", output_dir))
    image_list = [np.array(img) for img in image_list]
    import cv2
    height, width, layers = image_list[0].shape
    video_path_mp4 = os.path.join("render_result", output_dir, os.path.join(file_name+".mp4"))
    # cv2.VideoWriter creates a video writer object.
    # Parameters:
    # video_path_mp4: The path where the video file will be saved.
    # cv2.VideoWriter_fourcc(*'mp4v'): The codec used to compress the frames. 'mp4v' is a codec for .mp4 files.
    # 15: The number of frames per second (fps).
    # (width, height): The size of the video frames.
    video_mp4 = cv2.VideoWriter(video_path_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for img in image_list:
        video_mp4.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

    video_mp4.release()



def __drawFunc(frame_index = 0, image_list = []):
    # glClearColor(173.0/255, 216.0/255, 230.0/255, 0.0)
    glClearDepth(1.0)
    glPointSize(5)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)

    current_frame = glutGet(GLUT_ELAPSED_TIME)

    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    view = camera.get_view_matrix()

    # Extract the rotation matrix (upper-left 3x3)
    R = glm.mat3(view)

    # Extract the translation vector (last column)
    T = glm.vec3(view[3])

    # Calculate the camera position in world space
    viewPos =  glm.vec3(-glm.transpose(R) * T)


    shader_program.use()
    shader_program.set_matrix("projection", glm.value_ptr(projection))
    shader_program.set_matrix("view", glm.value_ptr(view))

    m = glm.mat4(1.0)
    m = glm.translate(m, grid_position[2])
    m = glm.rotate(m, glm.radians(0), grid_position[1][1])
    m = glm.scale(m, glm.vec3(5))
    shader_program.set_matrix("model", glm.value_ptr(m))
    # floor_color = glm.vec3(1.0, 1.0, 1.0)


    # glUniform3fv(glGetUniformLocation(shader_program.id, "fragColor"), 1, glm.value_ptr(floor_color))
    # draw_background()
    # background_texture = load_texture("background.png")

    floor_model.draw(shader_program, draw_type=GL_TRIANGLES)
    # shader_program.un_use()
    # shader_program.use()
    # grid_color = glm.vec3(0.0, 0.2, 0.3)
    # glUniform3fv(glGetUniformLocation(shader_program.id, "fragColor"), 1, glm.value_ptr(grid_color))
    # glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    # grid_model.draw(shader_program, draw_type=GL_LINES)
    # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    shader_program.un_use()
    glEnable(GL_DEPTH_TEST)

    robot_program.use()
    robot_program.set_matrix("projection", glm.value_ptr(projection))
    # Pass view_position (vec3) to the shader instead of the entire matrix
    glUniform3fv(glGetUniformLocation(robot_program.id, "viewPos"), 1, glm.value_ptr(viewPos))

    glUniform3fv(glGetUniformLocation(robot_program.id, "lightPos"), 1, light_pos)
    glUniformMatrix4fv(glGetUniformLocation(robot_program.id, "view"), 1, GL_FALSE, glm.value_ptr(view))




    m = glm.mat4(1.0)
    # m = glm.rotate(m, glm.radians(-90), glm.vec3(1, 0, 0))
    robot_program.set_matrix("model", glm.value_ptr(m))

    # # Define the light properties
    # light_direction = np.array([0.0, -1.0, -1.0], dtype=np.float32)  # Direction of light
    # light_ambient = np.array([0.1, 0.1, 0.1], dtype=np.float32)  # Ambient intensity
    # light_diffuse = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # Diffuse intensity
    # light_specular = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # Specular intensity
    #
    # # Set uniforms
    # glUniform3fv(glGetUniformLocation(shader_program, "lightDirection"), 1, light_direction)
    # glUniform3fv(glGetUniformLocation(shader_program, "lightAmbient"), 1, light_ambient)
    # glUniform3fv(glGetUniformLocation(shader_program, "lightDiffuse"), 1, light_diffuse)
    # glUniform3fv(glGetUniformLocation(shader_program, "lightSpecular"), 1, light_specular)

    robot_program.un_use()

    human_model.play_animation(robot_program, frame_index)

    #
    global last_frame
    last_frame = glutGet(GLUT_ELAPSED_TIME)
    global delta_time
    delta_time = (last_frame - current_frame)
    camera.process_keyboard(delta_time / 1000)
    if delta_time < 16:
        time.sleep((16 - delta_time) / 1000)
    calculate_FPS()
    global fps_count
    if fps_count == 100:
        fps_count = 0
        print('fps: %.2f' % _fps)
    fps_count += 1
    glutSwapBuffers()

    pixels = glReadPixels(0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(SCR_HEIGHT, SCR_WIDTH, 3)
    image = np.flipud(image)
    from PIL import Image
    img = Image.fromarray(image)
    image_list.append(img)
    glutPostRedisplay()

    error = glGetError()
    if error != GL_NO_ERROR:
        print("OpenGL Error:", error)
    return image

def __init_rend():
    glutInit()
    glutInitContextVersion(3, 3)
    glutInitContextProfile(GLUT_CORE_PROFILE)
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(SCR_WIDTH, SCR_HEIGHT)
    glutCreateWindow(b"demo")
    # glutSetCursor(GLUT_CURSOR_NONE)
    print(glGetString(GL_VERSION))

    global prevTicks
    prevTicks = glutGet(GLUT_ELAPSED_TIME)

    init()

    glutDisplayFunc(drawFunc)

if __name__ != '__main__':
    __init_rend()

if __name__ == "__main__":
    main()
