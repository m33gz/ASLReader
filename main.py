import pygame
import pygame.camera
from pygame.locals import *
import sys
import cv2
import numpy as np
from pygame.locals import *
import mediapipe as mp

# Initialize Pygame
pygame.init()
pygame.camera.init()

# Initialize Variables
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up Pygame window
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('HandTracker')

# Set up Pygame clock
clock = pygame.time.Clock()

# Set up Pygame colors
white = (255, 255, 255)

# Set up Mediapipe Hands
hands = mp_hands.Hands()

# Set up Pygame camera
pygame.camera.init()
camera = pygame.camera.Camera(pygame.camera.list_cameras()[0])
camera.start()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Get Pygame webcam image
    img = pygame.surfarray.array3d(camera.get_image())

    # Convert the BGR image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image with Mediapipe Hands
    results = hands.process(img_rgb)

    # Draw landmarks if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Convert the RGB image to a Pygame surface
    img_surface = pygame.surfarray.make_surface(np.flipud(img_rgb))



    # Display the image in the Pygame window
    screen.blit(img_surface, (0, 0))
    pygame.display.update()

    # Cap the frame rate
    clock.tick(30)