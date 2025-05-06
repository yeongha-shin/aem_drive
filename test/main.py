import numpy as np
import matplotlib.pyplot as plt
import pygame
import math

from car_model import Car2
from lane_following import CurvedRoad # if needed elsewhere

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

if __name__ == "__main__":
    # Set up screen and clock
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Vertical Car Sim - Multiple Cars")
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

    # Variables for deviation sampling
    total_deviation = 0.0
    deviation_timer = 0.0  # accumulates time until 1 second
    sampling_interval = 1.0  # seconds

    done = False
    
    while not done:
        # Delta time based on frame rate
        dt = clock.tick(60) / 1000.0  # seconds

        # --- Input handling
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
                elif event.key == pygame.K_UP:
                    car.release_down(1)

        # Also allow continuous key hold
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            car.accelerate(1)
        if keys[pygame.K_DOWN]:
            car.accelerate(-1)
        if keys[pygame.K_LEFT]:
            car.turn(-1)
        if keys[pygame.K_RIGHT]:
            car.turn(1)

        # --- Update simulation
        screen.fill(WHITE)
        drawRoad(screen)
        road.plotRoad(screen)

        car.update(dt)
        for c in [car2, car3, car4]:
            c.update(dt)

        updateSteering(screen, car)
        updateSpeedometer(screen, car)

        # --- Deviation sampling every second
        deviation_timer += dt
        if deviation_timer >= sampling_interval:
            deviation_value = road.deviation(car)
            total_deviation += abs(deviation_value)
            print(f"Sampled deviation: {deviation_value:.2f}, Total deviation: {total_deviation:.2f}")
            deviation_timer -= sampling_interval


        # --- Goal check
        if car.pose[1] < 50:
            print('Reached y=50, simulation done.')
            done = True
        
        pygame.display.flip()

    pygame.quit()
