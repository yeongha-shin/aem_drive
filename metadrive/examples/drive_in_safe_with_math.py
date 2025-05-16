#!/usr/bin/env python

import logging
import random
import time
import os

import numpy as np
import pandas as pd

from metadrive.constants import HELP_MESSAGE
from metadrive.envs.metadrive_env import MetaDriveEnv as ComplexEnv, get_condition_label, speak
from metadrive.component.vehicle_model.bicycle_model import BicycleModel
from metadrive.policy.env_input_policy import EnvInputPolicy

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, LineSegs, NodePath
from direct.showbase import ShowBaseGlobal

import logitech_steering_wheel as lsw

# Globals for prediction line and experiment tracking
prediction_path_node = None
marker_nodes = []

experimenter_id = "002"
log_file = "experiment_log.xlsx"
total_time = 0.8           # seconds ahead to predict trajectory
time_limit = 5             # seconds to answer math problem
CONDITION = get_condition_label()


def draw_prediction_path(points):
    """Draw colored trajectory line based on predicted points."""
    global prediction_path_node

    # remove previous trajectory if present
    if prediction_path_node:
        prediction_path_node.removeNode()

    segs = LineSegs()
    segs.setThickness(4.0)

    n = len(points)
    for i, (x, y) in enumerate(points):
        t = i / max(n - 1, 1)  # normalized position along path
        segs.setColor(1 - t, t, 0.0, 1.0)
        segs.drawTo(x, y, 0.1)

    # create and attach new line node
    prediction_path_node = NodePath(segs.create())
    prediction_path_node.setLightOff()
    prediction_path_node.setShaderOff()
    prediction_path_node.reparentTo(render)


def generate_math_problem():
    """Create a simple addition or subtraction problem."""
    a = random.randint(10, 99)
    b = random.randint(10, 99)
    op = random.choice(["+", "-"])
    result = eval(f"{a}{op}{b}")
    return f"{a} {op} {b} = ?", result


def set_new_options():
    """Pick two options (one correct, one near-miss) and reset timer."""
    global N_value, M_value, current_answer, problem_start_time

    correct = current_answer
    wrong = correct
    while wrong == correct:
        wrong = random.randint(correct - 3, correct + 3)

    if random.random() < 0.5:
        N_value, M_value = correct, wrong
    else:
        N_value, M_value = wrong, correct

    options_display.setText(f"L: {N_value}    R: {M_value}")
    problem_start_time = time.time()


def handle_input(key):
    """Check user input (n or m) against the correct answer and give feedback."""
    global waiting_for_answer, showing_feedback, last_feedback_time
    global feedback_text, current_answer, math_score

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
    # initialize environment and model
    env = ComplexEnv({
        "use_render": True,
        "manual_control": False,
        "agent_policy": EnvInputPolicy,
        "vehicle_config": {"show_navi_mark": False},
        "show_interface": False,
        "show_coordinates": False
    })
    bike_model = BicycleModel()

    try:
        env.reset()
        keys = ShowBaseGlobal.base.mouseWatcherNode

        # attempt wheel initialization
        try:
            handle = int(env.engine.win.get_window_handle().get_int_handle())
            lsw.initialize_with_window(True, handle)
            USE_WHEEL = lsw.is_connected(0)
        except:
            USE_WHEEL = False
        print("Logitech wheel detected:", USE_WHEEL)

        # on-screen displays
        speed_display = OnscreenText(text="Speed: 0 km/h", pos=(-0.75, -0.95),
                                     scale=0.07, fg=(1,1,1,1), align=TextNode.ARight,
                                     mayChange=True)
        offroad_warning = OnscreenText(text="WARNING: Approaching road edge!",
                                       pos=(0, 0.6), scale=0.1,
                                       fg=(1,0.3,0.3,1), align=TextNode.ACenter,
                                       mayChange=True)
        offroad_warning.hide()

        # experiment counters
        env.agent.crash_vehicle_count = 0
        env.agent.off_road_count = 0
        start_time = time.time()
        math_score = 0
        env.agent.expert_takeover = False

        # prepare first math problem
        current_problem, current_answer = generate_math_problem()
        last_feedback_time = None
        waiting_for_answer = True
        showing_feedback = False
        feedback_text = ""
        N_value = M_value = None

        math_display = OnscreenText(text=current_problem, pos=(-1.2, 0.8),
                                    scale=0.15, fg=(1,1,0,1),
                                    align=TextNode.ALeft, mayChange=True)
        options_display = OnscreenText(text="", pos=(-1.2, 0.5),
                                       scale=0.1, fg=(1,1,1,1),
                                       align=TextNode.ALeft, mayChange=True)
        result_display = OnscreenText(text="", pos=(-1.2, 0.4),
                                      scale=0.1, fg=(0,1,0,1),
                                      align=TextNode.ALeft, mayChange=True)
        set_new_options()

        # main loop
        while True:
            prev_takeover = env.current_track_agent.expert_takeover

            # read controls from wheel or keyboard
            if USE_WHEEL:
                lsw.update()
                state = lsw.get_state(0)
                lsw.play_damper_force(0, 80)

                if waiting_for_answer:
                    if state.rgbButtons[5]:
                        handle_input('n')
                    elif state.rgbButtons[4]:
                        handle_input('m')

                steering = -max(min(state.lX / 32767, 1.0), -1.0)
                throttle = max(min((32767 - state.lY) / 65534, 1.0), 0.0)
                brake = max(min((32767 - state.lRz) / 65534, 1.0), 0.0)
                action = [steering, max(min(throttle - brake, 1.0), -1.0)]
            else:
                steering = throttle = 0.0
                if keys.is_button_down("a"): steering += 1.0
                if keys.is_button_down("d"): steering -= 1.0
                if keys.is_button_down("w"): throttle += 1.0
                if keys.is_button_down("s"): throttle -= 1.0

                if waiting_for_answer:
                    if keys.is_button_down("n"): handle_input("n")
                    elif keys.is_button_down("m"): handle_input("m")

                action = [steering, throttle]

            o, r, tm, tc, info = env.step(action)

            # predict and draw future trajectory
            x, y = env.agent.position
            speed = env.agent.speed
            heading = env.agent.heading_theta
            vx, vy = env.agent.velocity

            if np.hypot(vx, vy) > 1e-3:
                vel_angle = np.arctan2(vy, vx)
                beta = (vel_angle - heading + np.pi) % (2*np.pi) - np.pi
            else:
                beta = 0.0

            bike_model.reset(x, y, speed, heading, beta)
            dt = 0.025
            steps = int(total_time / dt)
            traj = [(x, y)]
            for _ in range(steps):
                st = bike_model.predict(dt, [action[1], action[0]])
                traj.append((st["x"], st["y"]))
            draw_prediction_path(traj)

            # offroad warning logic
            try:
                warn = False
                for px, py in traj:
                    long, lat = env.agent.lane.local_coordinates((px, py))
                    w = env.agent.navigation.get_current_lane_width() / 2
                    if min(lat + w, w - lat) < 0.9:
                        warn = True
                        break
                (offroad_warning.show() if warn else offroad_warning.hide())
            except:
                offroad_warning.hide()

            # math timeout handling
            if waiting_for_answer and time.time() - problem_start_time > time_limit:
                result_display.setText("Time out!")
                result_display.setFg((1,0.5,0,1))
                last_feedback_time = time.time()
                waiting_for_answer = False
                showing_feedback = True

            # update speed display
            speed_kmh = np.linalg.norm(env.agent.velocity) * 3.6
            speed_display.setText(f"Speed: {int(speed_kmh)} km/h")

            # exit on arrival
            if info.get("arrive_dest", False):
                break

            # reset math UI after feedback
            if showing_feedback and time.time() - last_feedback_time > 0.5:
                current_problem, current_answer = generate_math_problem()
                math_display.setText(current_problem)
                result_display.setText("")
                set_new_options()
                waiting_for_answer = True
                showing_feedback = False

            env.render()

            # log takeover warning once
            if not prev_takeover and env.current_track_agent.expert_takeover:
                logging.warning("Auto-Drive may fail in some cases")

            # reset environment on crash/arrival
            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True

    finally:
        # collect and save results
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
            df_new = pd.concat([df_existing, df_new], ignore_index=True)
        df_new.to_excel(log_file, index=False)
        print("Results saved.")

        env.close()
        lsw.shutdown()
