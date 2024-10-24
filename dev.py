import pyray as pr
import numpy as np
import configs
from rubik import Rubik
from utils import generate_random_movements

pr.init_window(configs.window_w, configs.window_h, "Raylib [core] example - 3d camera")

rubik_cube = Rubik()

rotation_queue = generate_random_movements(100)
# Manual movements
# [
#     (-1.5707963267948966, np.array([0, 0, 1]), 2),
#     (-1.5707963267948966, np.array([0, 0, 1]), 2),
#     (-1.5707963267948966, np.array([0, 0, 1]), 0),
# ]


pr.set_target_fps(configs.fps)

while not pr.window_should_close():

    rotation_queue, _ = rubik_cube.handle_rotation(rotation_queue)

    pr.update_camera(configs.camera, pr.CameraMode.CAMERA_ORBITAL)

    pr.begin_drawing()
    pr.clear_background(pr.RAYWHITE)

    pr.begin_mode_3d(configs.camera)
    pr.draw_grid(20, 1.0)

    # Draw each cube of the Rubik's Cube
    for i, cube in enumerate(rubik_cube.cubes):
        for cube_part in cube:
            position = pr.Vector3(
                cube[0].center[0], cube[0].center[1], cube[0].center[2]
            )
            print(cube[0].center)
            pr.draw_model(cube_part.model, position, 2, cube_part.face_color)

    # position = pr.Vector3(piece.center[0], piece.center[1], piece.center[2])
    # pr.draw_model(piece.model, position, 2, pr.BLACK)

    pr.end_mode_3d()
    pr.end_drawing()

pr.close()
