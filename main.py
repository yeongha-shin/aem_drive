# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pygame
import math

from car_model import Car2
from lane_following import CurvedRoad

# Initialize pygame
pygame.init()

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

size = (600, 1400)  # 화면을 세로로 길게
PI = math.pi

def updateSteering(screen, car):
    pygame.draw.arc(screen, GREEN, [20, 20, 250, 200], PI / 4, 3 * PI / 4, 5)
    pygame.draw.arc(screen, RED, [20, 20, 250, 200], 3 * PI / 4, PI, 5)
    pygame.draw.arc(screen, RED, [20, 20, 250, 200], 0, PI / 4, 5)
    pygame.draw.circle(screen, BLACK, [145, 120], 20)

    x1 = 145 - 145
    y1 = 10 - 120
    x2 = x1 * math.cos(car.steering_angle) - y1 * math.sin(car.steering_angle)
    y2 = x1 * math.sin(car.steering_angle) + y1 * math.cos(car.steering_angle)
    x = x2 + 145
    y = y2 + 120
    pygame.draw.line(screen, BLACK, [x, y], [145, 120], 5)

def drawRoad(screen):
    pygame.draw.line(screen, BLACK, (300, 1400), (300, 0), 60)

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

    speed_text = font.render("Speed: " + str(car.speed / 5), True, BLACK)
    screen.blit(speed_text, [300, 60])

def gameLoop(action, car, screen):
    if action == 1 or action == 'a' or action == 'left':
        car.turn(-1)
    elif action == 2 or action == 'd' or action == 'right':
        car.turn(1)

def learningGameLoop():
    print('more code here')

class laneFollowingCar1(Car2):
    def __init__(self):
        super().__init__(RED, 300, 1300, screen)
        self.car = super().car
        self.car.constant_speed = True
        self.car.speed = 100

if __name__ == "__main__":
    t = 0

    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Vertical Car Sim - Multiple Cars")
    background = pygame.Surface(screen.get_size())
    background.fill((0, 0, 0))

    done = False
    clock = pygame.time.Clock()

    # --- Main car (user-controlled)
    car = Car2(RED, 300, 1300, screen)
    car.constant_speed = True
    car.speed = 50
    car.angle = -math.pi / 2  # 위쪽 방향

    # --- Other cars (AI cars)
    car2 = Car2(BLUE, 250, 1400, screen)    # 왼쪽 살짝
    car3 = Car2(GREEN, 350, 1500, screen)   # 오른쪽 살짝
    car4 = Car2(YELLOW, 300, 1600, screen)  # 중앙, 더 뒤

    for c in [car2, car3, car4]:
        c.constant_speed = True
        c.speed = 40  # 느리게
        c.angle = -math.pi / 2  # 다 위쪽

    # --- Road
    road = CurvedRoad(1200, 300, 1300, '45')

    screen.fill(WHITE)

    rate = 10

    while not done:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            car.accelerate(1)
        if keys[pygame.K_DOWN]:
            car.accelerate(-1)
        if keys[pygame.K_LEFT]:
            car.turn(-1)
        if keys[pygame.K_RIGHT]:
            car.turn(1)

        t += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    car.turn(-1)
                elif event.key == pygame.K_RIGHT:
                    car.turn(1)
                elif event.key == pygame.K_UP:
                    car.accelerate(1)
                elif event.key == pygame.K_DOWN:
                    car.accelerate(-1)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    car.release_down(-1)
                if event.key == pygame.K_UP:
                    car.release_down(1)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print("User pressed a mouse button")

        screen.fill(WHITE)

        # --- Draw everything
        drawRoad(screen)
        road.plotRoad(screen)

        car.update(1 / rate)
        for c in [car2, car3, car4]:
            c.update(1 / rate)

        updateSteering(screen, car)
        updateSpeedometer(screen, car)

        # --- Reward (only for main car)
        print(road.reward(car))

        # --- Goal check
        if car.pose[1] < 50:
            print('reached y=50')
            car.speed = 0
            done = True

        if t > 10000:
            car.speed = 0
            print('Time out!')
            done = True

        pygame.display.flip()
        clock.tick(rate)
