#!/usr/bin/env python
import logging
import random
import time

from metadrive.constants import HELP_MESSAGE
from metadrive.tests.test_functionality.test_object_collision_detection import ComplexEnv

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode
from direct.showbase.ShowBase import ShowBase

# 수학 문제 생성 함수
def generate_math_problem():
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    op = random.choice(["+", "-", "*"])
    result = eval(f"{a}{op}{b}")
    problem = f"{a} {op} {b} = ?"
    return problem, result

# 선택지 구성 함수
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
    options_display.setText(f"N: {N_value}    M: {M_value}")

# 키 입력 처리 함수
def handle_input(key):
    global waiting_for_answer, showing_feedback, last_feedback_time
    global feedback_text, current_problem, current_answer

    if not waiting_for_answer:
        return  # 이미 답했거나 피드백 표시 중이면 무시

    selected = N_value if key == 'n' else M_value
    if selected == current_answer:
        feedback_text = "Correct!"
        result_display.setFg((0, 1, 0, 1))  # 초록색
    else:
        feedback_text = "Wrong!"
        result_display.setFg((1, 0, 0, 1))  # 빨간색

    result_display.setText(feedback_text)
    last_feedback_time = time.time()
    waiting_for_answer = False
    showing_feedback = True

if __name__ == "__main__":
    env = ComplexEnv(dict(use_render=True, manual_control=True, vehicle_config={"show_navi_mark": False}))
    try:
        env.reset()
        print(HELP_MESSAGE)
        env.agent.expert_takeover = True

        # 상태 변수 초기화
        current_problem, current_answer = generate_math_problem()
        last_feedback_time = None
        waiting_for_answer = True
        showing_feedback = False
        feedback_text = ""
        N_value = None
        M_value = None

        # 수학 문제 표시
        math_display = OnscreenText(
            text=current_problem,
            pos=(-1.2, 0.8),
            scale=0.15,
            fg=(1, 1, 0, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

        # 선택지 표시
        options_display = OnscreenText(
            text="",
            pos=(-1.2, 0.6),
            scale=0.1,
            fg=(1, 1, 1, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

        # 정답/오답 피드백 표시
        result_display = OnscreenText(
            text="",
            pos=(-1.2, 0.4),
            scale=0.1,
            fg=(0, 1, 0, 1),
            align=TextNode.ALeft,
            mayChange=True
        )

        # 초기 옵션 설정
        set_new_options()

        # 키 입력 바인딩
        base.accept('n', handle_input, ['n'])
        base.accept('m', handle_input, ['m'])

        while True:
            previous_takeover = env.current_track_agent.expert_takeover
            o, r, tm, tc, info = env.step([0, 0])

            # 피드백 상태 처리
            if showing_feedback and time.time() - last_feedback_time > 3:
                current_problem, current_answer = generate_math_problem()
                math_display.setText(current_problem)
                result_display.setText("")
                set_new_options()
                waiting_for_answer = True
                showing_feedback = False

            # HUD 렌더링
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Total episode cost": env.episode_cost,
                    "Keyboard Control": "W,A,S,D",
                    "Answer with": "Press N or M",
                }
            )

            if not previous_takeover and env.current_track_agent.expert_takeover:
                logging.warning("Auto-Drive mode may fail to solve some scenarios due to distribution mismatch")

            if (tm or tc) and info["arrive_dest"]:
                env.reset()
                env.current_track_agent.expert_takeover = True

    finally:
        env.close()
