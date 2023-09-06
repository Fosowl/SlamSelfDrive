
import cv2 as cv
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class Renderer3D:
    def __init__(self) -> None:
        self.x_rot = 0
        self.y_rot = 0
        self.rotate_speed_x = 0.0
        self.rotate_speed_y = 0.0
        self.display = (800, 600)
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
    
    def draw_point(self, point, color=(0.0, 1.0, 0.0)):
        gColor3f(color)
        glVertex3f(point)
    
    def draw_lines(self, start, end, color=(0.0, 1.0, 0.0)):
        gColor3f(color)
        glVertex3f(start)
        glVertex3f(end)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    self.rotate_speed_y += 0.1  # X-axis rotation
                elif event.key == pygame.K_k:
                    self.rotate_speed_y -= 0.1  # X-axis rotation
                if event.key == pygame.K_j:
                    self.rotate_speed_x -= 0.1  # Y-axis rotation
                elif event.key == pygame.K_l:
                    self.rotate_speed_x += 0.1
        self.x_rot += self.rotate_speed_y
        self.y_rot += self.rotate_speed_x
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, 0.0)
        glRotatef(self.x_rot, 1, 0, 0)  # Rotate around X-axis
        glRotatef(self.y_rot, 0, 1, 0)  # Rotate around Y-axis
    
    def render(self):
        pygame.display.flip()
        pygame.time.wait(10)
    
    def render3dSpace(self, points):
        if points is None:
            return
        glBegin(GL_POINTS)
        for point in points:
            self.draw_point(point)
        glEnd()