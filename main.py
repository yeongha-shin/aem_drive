import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle, Circle
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
import imageio
from PIL import Image

# ===================================
# Color 정의
# ===================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# ===================================
# 보조 함수
# ===================================
def dampenSteering(angle, elasticity, delta):
    if angle == 0:
        return 0
    elif angle > 0:
        return max(angle - elasticity, 0)
    else:
        return min(angle + elasticity, 0)

def dampenSpeed(speed, velocity_dampening, delta):
    if speed == 0:
        return 0
    elif speed > 0:
        return max(speed - velocity_dampening * delta * (speed / 10), 0)
    else:
        return min(speed - velocity_dampening * delta * (speed / 10), 0)

def updateSpeedometer(screen, car):
    font = pygame.font.SysFont('Calibri', 25, True, False)
    x_base = 1400  # 오른쪽 공간 시작점

    if car.gear == "D":
        gear_text = font.render("Gear: Drive", True, BLACK)
    elif car.gear == "STOP":
        gear_text = font.render("Gear: Stopped", True, BLACK)
    elif car.gear == "R":
        gear_text = font.render("Gear: Reverse", True, BLACK)
    else:
        gear_text = font.render("Gear: unknown", True, BLACK)

    screen.blit(gear_text, (x_base, 40))
    speed_text = font.render(f"Speed: {int(car.speed)} km/h", True, BLACK)
    screen.blit(speed_text, (x_base, 80))

def draw_dynamic_obstacles_on_matplotlib(ax, scenario, car, current_time_step, distance_threshold=30):
    car_x, car_y = car.pose

    for obstacle in scenario.dynamic_obstacles:
        predicted_state = None
        for state in obstacle.prediction.trajectory.state_list:
            if state.time_step == current_time_step:
                predicted_state = state
                break

        if predicted_state:
            pos_x, pos_y = predicted_state.position
            distance = math.hypot(pos_x - car_x, pos_y - car_y)

            if distance <= distance_threshold:
                color = 'red'
                radius = 1.5  # 원의 반지름
                circle = Circle(
                    (pos_x, pos_y),
                    radius=radius,
                    edgecolor=color,
                    facecolor=color,
                    lw=2,
                    zorder=20
                )
                ax.add_patch(circle)




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
        self.width = 2
        self.length = 5
        self.maxSteer = math.pi
        self.acceleration_rate = 5
        self.speed_dampening = 0.1
        self.maxSpeed = 300
        self.delta = 1 / 60
        self.steering_elasticity = 5
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
        new_steering_angle = self.steering_angle + direction
        if new_steering_angle > self.maxSteer:
            self.steering_angle = self.maxSteer
        elif new_steering_angle < -self.maxSteer:
            self.steering_angle = -self.maxSteer
        else:
            self.steering_angle = new_steering_angle

    def update(self, delta):
        self.delta = delta
        self.angle += self.steering_angle * delta * self.speed / 20
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
            edgecolor='green',
            facecolor='green',
            lw=2,
            zorder=10
        )
        ax.add_patch(rect)

# ===================================
# Main Simulation
# ===================================
file_path = "./scenario/USA_Lanker-2_25_T-1.xml"
scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
max_time_step = max([obs.prediction.final_time_step for obs in scenario.dynamic_obstacles]) if scenario.dynamic_obstacles else 50

pygame.init()
screen = pygame.display.set_mode((2000, 1600))
pygame.display.set_caption('CommonRoad + Car Driving')
clock = pygame.time.Clock()

car = Car2(color='green', x=0, y=0)
car.constant_speed = True
car.speed = 5
car.angle = math.radians(60)

current_time_step = 0
frames = []  # gif 저장용

running = True
while running:
    dt = clock.tick(30) / 1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        car.accelerate(0.2)
    if keys[pygame.K_DOWN]:
        car.accelerate(-0.2)
    if keys[pygame.K_LEFT]:
        car.turn(1)
    if keys[pygame.K_RIGHT]:
        car.turn(-1)

    fig, ax = plt.subplots(figsize=(40, 30))
    renderer = MPRenderer(ax=ax)
    renderer.draw_params.show_labels = False
    renderer.draw_params.time_begin = current_time_step
    renderer.draw_params.time_end = current_time_step
    scenario.draw(renderer)
    planning_problem_set.draw(renderer)
    renderer.render()

    car.update(dt)
    car.draw_on_matplotlib(ax)
    draw_dynamic_obstacles_on_matplotlib(ax, scenario, car, current_time_step, distance_threshold=10)

    canvas = FigureCanvas(fig)
    canvas.draw()
    renderer_backend = canvas.get_renderer()
    raw_data = renderer_backend.buffer_rgba()
    size = canvas.get_width_height()
    background = pygame.image.frombuffer(raw_data, size, "RGBA")
    background = pygame.transform.scale(background, (2000, 1600))

    plt.close(fig)

    screen.blit(background, (0, 0))
    updateSpeedometer(screen, car)
    pygame.display.flip()

    # 현재 프레임 저장
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    image = Image.fromarray(frame)
    frames.append(image)

    current_time_step += 1
    if current_time_step >= max_time_step:
        running = False

pygame.quit()

# ===== GIF 저장 =====
frames[0].save(
    './results/simulation.gif',
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=33,
    loop=0
)
print("GIF 저장 완료: simulation.gif")
