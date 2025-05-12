#!/usr/bin/env python
import logging
import random
import time

from metadrive.constants import HELP_MESSAGE
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv as ComplexEnv


from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
from direct.showbase.ShowBase import ShowBase

import logitech_steering_wheel as lsw

from metadrive.envs.metadrive_env import get_condition_label

import pandas as pd
import os

experimenter_id = "001"  # ← Set this per participant
log_file = "experiment_log.xlsx"

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
            vehicle_config={"show_navi_mark": False}
        ))


    try:
        env.reset()
        env.agent.crash_vehicle_count = 0
        env.agent.off_road_count = 0
        start_time = time.time()
        math_score = 0
        #print(HELP_MESSAGE)
        env.agent.expert_takeover = False
        lsw.initialize_with_window(True, int(env.engine.win.get_window_handle().get_int_handle()))


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
            pos=(-1.2, 0.6),
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

            lsw.update()
            state = lsw.get_state(0)

            lsw.play_damper_force(0, 80)  # value: 0–100

            # Check wheel buttons manually
            if waiting_for_answer:
                if state.rgbButtons[5]:
                    handle_input('n')  # button 4 = N
                elif state.rgbButtons[4]:
                    handle_input('m')  # button 5 = M

            steering_raw = state.lX
            throttle_raw = state.lY
            brake_raw = state.lRz

            steering = -max(min(steering_raw / 32767, 1.0), -1.0)
            throttle = max(min((32767 - throttle_raw) / 65534, 1.0), 0.0)
            brake = max(min((32767 - brake_raw) / 65534, 1.0), 0.0)

            throttle_final = throttle - brake
            throttle_final = max(min(throttle_final, 1.0), -1.0)

            action = [steering, throttle_final]

            o, r, tm, tc, info = env.step(action)

            if info.get("arrive_dest", False):
                print(f"Arrived! Total penalty: {env.episode_cost}")
                break  # exit simulation



            if showing_feedback and time.time() - last_feedback_time > 3:
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


