#!/usr/bin/env python
import logging
import random
import time

from metadrive.constants import HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv as ComplexEnv



from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
from direct.showbase.ShowBase import ShowBase

import logitech_steering_wheel as lsw

from metadrive.envs.metadrive_env import get_condition_label, speak

from direct.gui.OnscreenText import OnscreenText
import numpy as np
from panda3d.core import TextNode


import pandas as pd
import os

from direct.showbase import ShowBaseGlobal


experimenter_id = "002"  # ← Set this per participant
log_file = "experiment_log.xlsx"

OFFROAD_WARNING_MS = 500  # milliseconds

CONDITION = get_condition_label()

def generate_math_problem():
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    op = random.choice(["+", "-", "*"])
    result = eval(f"{a}{op}{b}")
    problem = f"{a} {op} {b} = ?"
    return problem, result

def set_new_options():
    global N_value, M_value, current_answer
    correct = current_answer
    wrong = correct
    while wrong == correct:
        wrong = random.randint(correct - 3, correct + 3)
    if random.random() < 0.5:
        N_value, M_value = correct, wrong
    else:
        N_value, M_value = wrong, correct
    options_display.setText(f"L: {N_value}    R: {M_value}")

def handle_input(key):
    global waiting_for_answer, showing_feedback, last_feedback_time
    global feedback_text, current_problem, current_answer
    global math_score

    if not waiting_for_answer:
        return

    selected = N_value if key == 'n' else M_value
    if selected == current_answer:
        feedback_text = "Correct!"
        math_score += 1
        result_display.setFg((0, 1, 0, 1))
    else:
        feedback_text = "Wrong!"
        math_score -= 1
        result_display.setFg((1, 0, 0, 1))

    result_display.setText(feedback_text)
    last_feedback_time = time.time()
    waiting_for_answer = False
    showing_feedback = True

if __name__ == "__main__":
    from metadrive.policy.env_input_policy import EnvInputPolicy

    env = ComplexEnv(dict(
            use_render=True,
            manual_control=False,         # disables MetaDrive internal control
            agent_policy=EnvInputPolicy,  # allows external control via env.step()
            vehicle_config={"show_navi_mark": False},
            show_interface=False,
            show_coordinates=False
        ))


    try:
        env.reset()

        keys = ShowBaseGlobal.base.mouseWatcherNode

        try:
            lsw.initialize_with_window(True, int(env.engine.win.get_window_handle().get_int_handle()))
            USE_WHEEL = lsw.is_connected(0)
        except:
            USE_WHEEL = False
        print("Logitech wheel detected:", USE_WHEEL)

        '''
        cam = env.engine.main_camera
        cam.camera_dist = 0.5  # means exactly at the center of the car
        cam.chase_camera_height = 1.5  # roughly driver's eye level
        cam.camera_smooth = False  # disable smoothing for instant camera response
        '''

        speed_display = OnscreenText(
            text="Speed: 0 km/h",
            pos=(-0.75, -0.95),               # (X, Y) position on screen (bottom-left)
            scale=0.07,                    # size of the font
            fg=(1, 1, 1, 1),               # white color
            align=TextNode.ARight,        # align text to the right
            mayChange=True                # allows updates
        )

        debug_display = OnscreenText(
            text="",
            pos=(0.5, -0.5),
            scale=0.05,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

        offroad_warning = OnscreenText(
            text="WARNING: Approaching road edge!",
            pos=(0, 0.6),
            scale=0.1,
            fg=(1, 0.3, 0.3, 1),
            align=TextNode.ACenter,
            mayChange=True
        )
        offroad_warning.hide()


        env.agent.crash_vehicle_count = 0
        env.agent.off_road_count = 0
        start_time = time.time()
        math_score = 0
        #print(HELP_MESSAGE)
        env.agent.expert_takeover = False


        current_problem, current_answer = generate_math_problem()
        last_feedback_time = None
        waiting_for_answer = True
        showing_feedback = False
        feedback_text = ""
        N_value = None
        M_value = None


        math_display = OnscreenText(
            text=current_problem,
            pos=(-1.2, 0.8),
            scale=0.15,
            fg=(1, 1, 0, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

        options_display = OnscreenText(
            text="",
            pos=(-1.2, 0.5),
            scale=0.1,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

        result_display = OnscreenText(
            text="",
            pos=(-1.2, 0.4),
            scale=0.1,
            fg=(0, 1, 0, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

        set_new_options()

        #base.accept('n', handle_input, ['n'])
        #base.accept('m', handle_input, ['m'])

        while True:
            previous_takeover = env.current_track_agent.expert_takeover

            if USE_WHEEL:
                lsw.update()
                state = lsw.get_state(0)
                lsw.play_damper_force(0, 80)

                # Math input from wheel buttons
                if waiting_for_answer:
                    if state.rgbButtons[5]:
                        handle_input('n')
                    elif state.rgbButtons[4]:
                        handle_input('m')

                steering_raw = state.lX
                throttle_raw = state.lY
                brake_raw = state.lRz

                steering = -max(min(steering_raw / 32767, 1.0), -1.0)
                throttle = max(min((32767 - throttle_raw) / 65534, 1.0), 0.0)
                brake = max(min((32767 - brake_raw) / 65534, 1.0), 0.0)

                throttle_final = throttle - brake
                throttle_final = max(min(throttle_final, 1.0), -1.0)

                action = [steering, throttle_final]

            else:
                steering = 0.0
                throttle = 0.0

                if keys.is_button_down("a"):
                    steering += 1.0
                if keys.is_button_down("d"):
                    steering -= 1.0
                if keys.is_button_down("w"):
                    throttle += 1.0
                if keys.is_button_down("s"):
                    throttle -= 1.0

                # Math input from keyboard
                if waiting_for_answer:
                    if keys.is_button_down("n"):
                        handle_input("n")
                    elif keys.is_button_down("m"):
                        handle_input("m")

                action = [steering, throttle]


            o, r, tm, tc, info = env.step(action)

            try:
                pos = env.agent.position
                long, lat = env.agent.lane.local_coordinates(pos)
                heading = env.agent.lane.heading_theta_at(long)
                lane_width = env.agent.navigation.get_current_lane_width()

                # Edge distances (robust)
                left_edge = -lane_width / 2
                right_edge = lane_width / 2
                left_dist = lat - left_edge
                right_dist = right_edge - lat
                min_dist_to_edge = min(left_dist, right_dist)
                is_off_road = min_dist_to_edge < 0.9

                velocity = env.agent.velocity
                lane_normal = np.array([-np.sin(heading), np.cos(heading)])
                v_lateral = np.dot(velocity, lane_normal)

                if abs(v_lateral) < 1e-2:
                    t_offroad = None
                    side = None
                elif v_lateral > 0:
                    t_offroad = (left_dist - 0.9) / v_lateral if left_dist > 0.9 else 0
                    side = "left"
                else:
                    t_offroad = (right_dist - 0.9) / abs(v_lateral) if right_dist > 0.9 else 0
                    side = "right"

            except:
                long = lat = heading = lane_width = left_dist = right_dist = min_dist_to_edge = v_lateral = 0
                t_offroad = None
                side = None
                is_off_road = False

            if t_offroad is None:
                t_display = "∞"
            elif t_offroad <= 0:
                t_display = "0.00s (now)"
            else:
                t_display = f"{t_offroad:.2f}s ({side})"

            debug_display.setFg((1, 0, 0, 1) if is_off_road else (1, 1, 1, 1))
            debug_display.setText(
                f"Pos: ({pos[0]:.1f}, {pos[1]:.1f})\n"
                f"Lane: long={long:.1f}, lat={lat:.1f}\n"
                f"L dist: {left_dist:.2f}  R dist: {right_dist:.2f}  (min={min_dist_to_edge:.2f})\n"
                f"Heading: {np.degrees(heading):.1f}°\n"
                f"v_lat: {v_lateral:.2f} m/s   t_off: {t_display}"
            )

            if t_offroad is not None and 0 < t_offroad * 1000 < OFFROAD_WARNING_MS:
                offroad_warning.show()
            else:
                offroad_warning.hide()



            speed = np.linalg.norm(env.agent.velocity) * 3.6  # m/s → km/h
            speed_display.setText(f"Speed: {int(speed)} km/h")


            if info.get("arrive_dest", False):
                #print(f"Arrived! Total penalty: {env.episode_cost}")
                break  # exit simulation



            if showing_feedback and time.time() - last_feedback_time > 0.5:
                current_problem, current_answer = generate_math_problem()
                math_display.setText(current_problem)
                result_display.setText("")
                set_new_options()
                waiting_for_answer = True
                showing_feedback = False

            env.render()
            '''
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Total episode cost": env.episode_cost,
                    "Keyboard Control": "W,A,S,D",
                    "Answer with": "Press N or M",
                }
            )
            '''

            if not previous_takeover and env.current_track_agent.expert_takeover:
                logging.warning("Auto-Drive mode may fail to solve some scenarios due to distribution mismatch")

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True

    finally:
        # === Save experiment results ===
        completion_time = round(time.time() - start_time, 2)

        results = {
            "experimenter_id": experimenter_id,
            "condition": CONDITION,
            "math_score": math_score,
            "num_collisions": env.agent.crash_vehicle_count,
            "num_offroad_events": env.agent.off_road_count,
            "completion_time": completion_time
        }

        df_new = pd.DataFrame([results])

        if os.path.exists(log_file):
            df_existing = pd.read_excel(log_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_excel(log_file, index=False)
        print("Results saved.")

        env.close()
        lsw.shutdown()
 



