
import cv2 as cv
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GL import *

class Renderer3D:
    def __init__(self) -> None:
        self.x_rot = 0
        self.y_rot = 0
        self.freeze = False
        self.count = 0
        self.translate_x = 0.0
        self.translate_y = 0.0
        self.scale_factor = 1.0
        self.sensitivity = 0.2
        self.display = (800, 600)
        pygame.init()
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
    
    def is_freeze(self):
        return self.freeze
    
    def handle_events(self):
        self.count += 1
        print("handling events", self.count)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                self.freeze = True
                if event.key == pygame.K_j:
                    self.translate_x -= 10  # Y-axis rotation
                elif event.key == pygame.K_l:
                    self.translate_x += 10
                if event.key == pygame.K_i:
                    self.translate_y -= 10  # Y-axis rotation
                elif event.key == pygame.K_k:
                    self.translate_y += 10
                if event.key == pygame.K_o:
                    self.scale_factor *= 10
                elif event.key == pygame.K_u:
                    self.scale_factor *= 0.1
        R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0]]) 
        t = np.array([self.translate_x, self.translate_y, 0.0])
        stack_r = np.vstack((R, [0, 0, 0]))
        stack_t = np.hstack((t.T, [1]))
        transformation_matrix = np.column_stack((stack_r, stack_t))
        #glLoadIdentity()
        glTranslatef(t[0], t[1], t[2])
        glMultMatrixf(transformation_matrix.T)
        glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
    
    def draw_point(self, point, color=(0.0, 1.0, 0.0)):
        glColor3f(color[0], color[1], color[2])
        glVertex3f(point[0], point[1], point[2])

    def draw_lines(self, start, end, color=(0.0, 1.0, 0.0)):
        glColor3f(color)
        glVertex3f(start)
        glVertex3f(end)

    def render(self):
        pygame.display.flip()
        pygame.time.wait(10)
    
    def select_view(self, view="fpv"):
        glMatrixMode(GL_MODELVIEW)
        if view == "fpv":
            R = pose['R'] # rotation matrix
            t = pose['t'] # translation vector
        else:
            R = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0]]) 
            t = np.array([0, 0, 0])
        stack_r = np.vstack((R, [0, 0, 0]))
        stack_t = np.hstack((t.T[0], [1]))
        transformation_matrix = np.column_stack((stack_r, stack_t))
        glMultMatrixf(transformation_matrix.T)
        glTranslatef(-t[0], -t[1], -t[2])
    
    def render3dSpace(self, points, pose, camera_matrix):
        assert len(pose['R']) == 3 and len(pose['R'][0]) == 3, "Rotation matrix should be 3x3"
        assert len(pose['t']) == 3, "Translation vector should be 1x3"
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if points is None:
            return
        glBegin(GL_POINTS)
        for point in points:
            self.draw_point(point)
        glEnd()