import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer

# ===================================
# Color 정의
# ===================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# ===================================
# 보조 함수 정의
# ===================================
def dampenSteering(angle, elasticity, delta):
    if angle == 0:
        return 0
    elif angle > 0:
        new_angle = angle - elasticity
        return max(new_angle, 0)
    elif angle < 0:
        new_angle = angle + elasticity
        return min(new_angle, 0)

def dampenSpeed(speed, velocity_dampening, delta):
    if speed == 0:
        return 0
    elif speed > 0:
        new_speed = speed - velocity_dampening * delta * (speed / 10)
        return max(new_speed, 0)
    elif speed < 0:
        new_speed = speed - velocity_dampening * delta * (speed / 10)
        return min(new_speed, 0)

def updateSpeedometer(screen, car):
    font = pygame.font.SysFont('Calibri', 25, True, False)
    if car.gear == "D":
        gear_text = font.render("Gear: Drive", True, BLACK)
    elif car.gear == "STOP":
        gear_text = font.render("Gear: Stopped", True, BLACK)
    elif car.gear == "R":
        gear_text = font.render("Gear: Reverse", True, BLACK)
    else:
        gear_text = font.render("Gear: unknown", True, BLACK)

    screen.blit(gear_text, [300, 40])
    speed_text = font.render("Speed: " + str(int(car.speed / 5)) + " km/h", True, BLACK)
    screen.blit(speed_text, [300, 60])

def updateOtherVehiclesInfo(screen, scenario, current_time_step):
    font = pygame.font.SysFont('Calibri', 20, True, False)
    start_y = 100
    spacing = 30
    for idx, obstacle in enumerate(scenario.dynamic_obstacles):
        predicted_state = None
        for state in obstacle.prediction.trajectory.state_list:
            if state.time_step == current_time_step:
                predicted_state = state
                break
        if predicted_state:
            pos_x = predicted_state.position[0]
            pos_y = predicted_state.position[1]
            speed = predicted_state.velocity
            info_text = f"ID {obstacle.obstacle_id}: ({int(pos_x)}, {int(pos_y)}), {int(speed)} m/s"
            text_surface = font.render(info_text, True, BLUE)
            screen.blit(text_surface, (50, start_y + idx * spacing))

# ===================================
# Car2 클래스
# ===================================
class Car2():
    def __init__(self, color, x, y, speed=0, angle=0):
        self.color = color
        self.vel = [0, 0]
        self.speed = speed
        self.angle = angle
        self.pose = [x, y]
        self.width = 2  # 차량 폭 (m)
        self.length = 5  # 차량 길이 (m)
        self.maxSteer = math.pi / 3
        self.acceleration_rate = 5
        self.speed_dampening = 0.1
        self.maxSpeed = 300
        self.delta = 1 / 60
        self.steering_elasticity = 5 / 60
        self.gear = "STOP"
        self.constant_speed = False
        self.steering_angle = 0

    def accelerate(self, dv):
        if self.gear == "STOP":
            if dv > 0:
                self.gear = "D"
            elif dv < 0:
                self.gear = "R"

        if self.gear == "D":
            self.speed += self.acceleration_rate * dv
            self.speed = min(self.speed, self.maxSpeed)
            if self.speed <= 0:
                self.speed = 0
        elif self.gear == "R":
            self.speed += self.acceleration_rate * dv
            self.speed = max(self.speed, -self.maxSpeed)
            if self.speed >= 0:
                self.speed = 0

    def turn(self, direction):
        new_steering_angle = self.steering_angle + direction * (math.pi / 20)
        if new_steering_angle > self.maxSteer:
            self.steering_angle = self.maxSteer
        elif new_steering_angle < -self.maxSteer:
            self.steering_angle = -self.maxSteer
        else:
            self.steering_angle = new_steering_angle

    def update(self, delta):
        self.delta = delta
        self.angle += self.steering_angle * delta * self.speed / 100
        self.vel[0] = math.cos(self.angle) * self.speed
        self.vel[1] = math.sin(self.angle) * self.speed
        self.pose[0] += self.vel[0] * delta
        self.pose[1] += self.vel[1] * delta
        self.steering_angle = dampenSteering(self.steering_angle, self.steering_elasticity, delta)
        if not self.constant_speed:
            self.speed = dampenSpeed(self.speed, self.speed_dampening, delta)

    def draw_on_matplotlib(self, ax):
        rect = Rectangle(
            (self.pose[0] - self.length/2, self.pose[1] - self.width/2),
            self.length, self.width,
            angle=np.degrees(self.angle),
            edgecolor='red',
            facecolor='red',
            lw=2,
            zorder=10
        )
        ax.add_patch(rect)

# ===================================
# Main Simulation
# ===================================
# 시나리오 로드
file_path = "./scenario/USA_Lanker-2_25_T-1.xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

max_time_step = max([obs.prediction.final_time_step for obs in scenario.dynamic_obstacles]) if scenario.dynamic_obstacles else 50

# pygame 초기화
pygame.init()
screen = pygame.display.set_mode((3200, 1600))
pygame.display.set_caption('CommonRoad + Car Driving')
clock = pygame.time.Clock()

# Car2 생성
car = Car2(color='red', x=0, y=0)
car.constant_speed = True
car.speed = 10
car.angle = math.radians(60)

# 시뮬레이션 시간 관리
current_time_step = 0

running = True
while running:
    dt = clock.tick(30) / 1000  # 초 단위

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 키 입력 처리
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        car.accelerate(1)
    if keys[pygame.K_DOWN]:
        car.accelerate(-1)
    if keys[pygame.K_LEFT]:
        car.turn(5)
    if keys[pygame.K_RIGHT]:
        car.turn(-5)

    # ----- CommonRoad 지도 + Car2 갱신 -----
    fig, ax = plt.subplots(figsize=(40, 20))
    renderer = MPRenderer(ax=ax)
    renderer.draw_params.show_labels = False
    renderer.draw_params.time_begin = current_time_step
    renderer.draw_params.time_end = current_time_step
    scenario.draw(renderer)
    planning_problem_set.draw(renderer)
    renderer.render()

    # Car2 업데이트 및 matplotlib 위에 직접 그리기
    car.update(dt)
    car.draw_on_matplotlib(ax)

    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer_backend = canvas.get_renderer()
    raw_data = renderer_backend.buffer_rgba()
    size = canvas.get_width_height()
    background = pygame.image.frombuffer(raw_data, size, "RGBA")
    background = pygame.transform.scale(background, (3200, 1600))

    plt.close(fig)

    # ------------------------------------------------------

    # 화면 그리기
    screen.blit(background, (0, 0))
    updateSpeedometer(screen, car)
    updateOtherVehiclesInfo(screen, scenario, current_time_step)
    pygame.display.flip()

    # 타임스텝 업데이트
    current_time_step += 1
    if current_time_step >= max_time_step:
        current_time_step = max_time_step

pygame.quit()
