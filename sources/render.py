
import cv2 as cv
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GL import *
import copy

class Renderer3D:
    def __init__(self) -> None:
        self.x_rot = 1.0
        self.y_rot = 1.0
        self.freeze = False
        self.translate_x = 0.0
        self.translate_y = 0.0
        self.scale_factor = 1.0
        self.sensitivity = 0.2
        self.display = (800, 600)
        self.background = None
        self.points_stack = []
        self.trajectory = []
        self.updated_position = [0, 0, 0]
        pygame.init()
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)
    
    def is_freeze(self):
        return self.freeze

    def draw_plane(self):
        # Define the vertices for the plane (adjust size and position as needed)
        plane_vertices = [
            (-10.0, 0.0, -10.0),
            (10.0, 0.0, -10.0),
            (10.0, 0.0, 10.0),
            (-10.0, 0.0, 10.0)
        ]
        plane_indices = [0, 1, 2, 0, 2, 3]
        plane_vertices = np.array(plane_vertices, dtype=np.float32)
        plane_indices = np.array(plane_indices, dtype=np.uint32)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, plane_vertices.nbytes, plane_vertices, GL_STATIC_DRAW)
        ibo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, plane_indices.nbytes, plane_indices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo)
        glColor3f(1.0, 1.0, 1.0)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glEnableClientState(GL_VERTEX_ARRAY)
        glDrawElements(GL_TRIANGLES, len(plane_indices), GL_UNSIGNED_INT, None)
        glDisableClientState(GL_VERTEX_ARRAY)

    def ready(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.freeze = not self.freeze
                if event.key == pygame.K_l:
                    self.translate_x -= 0.05
                elif event.key == pygame.K_j:
                    self.translate_x += 0.05
                if event.key == pygame.K_i:
                    self.translate_y -= 0.05
                elif event.key == pygame.K_k:
                    self.translate_y += 0.05
                if event.key == pygame.K_o:
                    self.scale_factor *= 1.3
                elif event.key == pygame.K_u:
                    self.scale_factor *= 0.7
                if event.key == pygame.K_z:
                    self.y_rot *= 1.2
                elif event.key == pygame.K_s:
                    self.y_rot *= 0.8
                if event.key == pygame.K_q:
                    self.x_rot *= 1.2
                elif event.key == pygame.K_d:
                    self.x_rot *= 0.8
        R = np.array([[-self.x_rot, 0, 0],
                      [0, 0, self.y_rot],
                      [0, 1, 0]]) 
        t = np.array([self.translate_x, self.translate_y, 0.0])
        stack_r = np.vstack((R, [0, 0, 0]))
        stack_t = np.hstack((t.T, [1]))
        transformation_matrix = np.column_stack((stack_r, stack_t))
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(t[0], t[1], t[2])
        glMultMatrixf(transformation_matrix.T)
        glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
    
    def draw_point(self, point, color=(0.0, 1.0, 0.0)):
        glColor3f(color[0], color[1], color[2])
        glVertex3f(point[0], point[1], point[2])

    def draw_lines(self, start, end, color=(0.0, 1.0, 1.0)):
        glColor3f(color[0], color[1], color[2])
        glVertex3f(start[0], start[1], start[2])
        glVertex3f(end[0], end[1], end[2])

    def render(self, milliseconds=10):
        pygame.display.flip()
        pygame.time.wait(milliseconds)
    
    """
    def select_view(self, R, t, view="top"):
        if view == "top":
            R = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, 1, 0]]) 
            t = np.array([0, 0, 0])
        stack_r = np.vstack((R, [0, 0, 0]))
        stack_t = np.hstack((t.T[0], [1]))
        transformation_matrix = np.column_stack((stack_r, stack_t))
        glMultMatrixf(transformation_matrix.T)
        glTranslatef(-t[0], -t[1], -t[2])
    """
    
    def render3dSpace(self, points, pose, camera_matrix):
        assert len(pose['R']) == 3 and len(pose['R'][0]) == 3, "Rotation matrix should be 3x3"
        assert len(pose['t']) == 3, "Translation vector should be 1x3"
        if points is None:
            return
        self.updated_position[0] += pose['t'][0]
        self.updated_position[1] += pose['t'][1]
        self.updated_position[2] = 0
        for point in points:
            point[0] = point[0] + self.updated_position[0] 
            point[1] = point[1] + self.updated_position[1] 
            point[2] = point[2] + self.updated_position[2] 
            self.points_stack.append(point)
        upd = copy.deepcopy(self.updated_position)
        self.trajectory.append(upd)
        glBegin(GL_POINTS)
        for point in self.points_stack:
            self.draw_point(point, color=(0.0, 1.0, 0.0))
        for point in self.trajectory:
            self.draw_point(point, color=(1.0, 0.0, 0.0))
        glEnd()