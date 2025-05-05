# import os
# import cv2
# import numpy as np
# from metadrive import MetaDriveEnv
# from metadrive.envs.scenario_env import ScenarioEnv
# from metadrive.utils.draw_top_down_map import draw_top_down_map
#
# # 폴더 생성
# os.makedirs("map_images", exist_ok=True)
#
# if __name__ == '__main__':
#     env = MetaDriveEnv(config=dict(num_scenarios=100, map=7, start_seed=0))
#     print("We are going to save 6 maps! 3 for PG maps, 3 for real world ones!")
#
#     count = 0
#     for i in range(2):
#         if i == 1:
#             env.close()
#             env = ScenarioEnv(dict(start_scenario_index=0, num_scenarios=3))
#         for j in range(3):
#             count += 1
#             env.reset(seed=j)
#             map_img = draw_top_down_map(env.current_map)  # numpy array (H, W)
#
#             # uint8로 변환 (OpenCV는 float32를 바로 저장 못함)
#             map_img_uint8 = (255 * map_img).astype(np.uint8)
#
#             filename = f"map_images/map_{count:02d}.png"
#             cv2.imwrite(filename, map_img_uint8)
#             print(f"Saved {filename}")
#     env.close()


import os
import cv2
import numpy as np
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
from metadrive.component.sensors.rgb_camera import RGBCamera  # 만약 config에 따라 필요하면 사용

# 폴더 생성
os.makedirs("map_images", exist_ok=True)

if __name__ == '__main__':
    # 원래 코드에서 사용한 config 복사
    config = dict(
        use_render=False,  # 이미지 저장이 목적이므로 렌더링 불필요
        manual_control=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=True, show_navi_mark=False, show_line_to_navi_mark=False),
        map=4,  # seven block
        start_seed=10,
    )

    env = MetaDriveEnv(config)
    env.reset(seed=21)

    # 맵 이미지 생성
    map_img = draw_top_down_map(env.current_map)
    map_img_uint8 = (255 * map_img).astype(np.uint8)

    filename = "map_images/map_from_config.png"
    cv2.imwrite(filename, map_img_uint8)
    print(f"Saved {filename}")

    env.close()
