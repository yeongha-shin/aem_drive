import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

# 설정
file_path = "./scenario/USA_Lanker-1_1_T-1.xml"  # 시나리오 경로
max_time_steps = 40  # 몇 개 time step까지 애니메이션 할지

# 시나리오 로드
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# matplotlib figure 생성
fig, ax = plt.subplots(figsize=(25, 10))


# update 함수 정의
def update(frame):
    ax.clear()

    # 새로운 Renderer 만들고 설정
    rnd = MPRenderer(ax=ax)
    rnd.draw_params.time_begin = frame
    rnd.draw_params.time_end = frame
    rnd.draw_params.show_labels = False

    scenario.draw(rnd)
    planning_problem_set.draw(rnd)

    rnd.render()

    ax.set_title(f"Time Step: {frame}")


# FuncAnimation으로 애니메이션 생성
anim = FuncAnimation(
    fig,
    update,
    frames=max_time_steps,  # 0 ~ max_time_steps-1
    interval=500,  # frame 간 시간 (ms) - 500ms == 0.5초
    repeat=False  # 다 끝나면 반복하지 않음
)

plt.show()
