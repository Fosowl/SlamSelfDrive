
"""
3D rendering with OpenGL
"""

import cv2 as cv
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL import *
import copy

class Renderer3D:
    def __init__(self) -> None:
        self.x_rot = 0.0
        self.y_rot = 0.0
        self.z_rot = 0.0
        self.freeze = False
        self.translate_x = 0.0
        self.translate_y = 0.0
        self.scale_factor = 0.5
        self.sensitivity = 0.2
        self.display = (800, 600)
        self.background = None
        self.points_stack = []
        self.trajectory = []
        pygame.init()
        pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        #glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH)
        gluPerspective(45, (self.display[0] / self.display[1]), 0.1, 50.0)
        glTranslatef(0.0, 0.0, 0)
    
    def is_freeze(self):
        """
        return whenever to freeze the world view to current 3D points
        """
        return self.freeze
    
    def key_update(self, key):
        """
        handle keys input
        """
        if key == pygame.K_l:
            self.translate_x -= 0.05
        elif key == pygame.K_j:
            self.translate_x += 0.05
        if key == pygame.K_i:
            self.translate_y -= 0.05
        elif key == pygame.K_k:
            self.translate_y += 0.05
        if key == pygame.K_o:
            self.scale_factor *= 1.3
        elif key == pygame.K_u:
            self.scale_factor *= 0.7
        if key == pygame.K_s:
            self.y_rot += 5
        elif key == pygame.K_f:
            self.y_rot -= 5
        if key == pygame.K_e:
            self.x_rot += 5
        elif key == pygame.K_d:
            self.x_rot -= 5
        if key == pygame.K_z:
            self.z_rot -= 5
        elif key == pygame.K_r:
            self.z_rot += 5

    def rotation_x(self, angle):
        """
        angle on x axis to rotation matrix
        """
        angle = np.radians(angle)
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    
    def rotation_y(self, angle):
        """
        angle on y axis to rotation matrix
        """
        angle = np.radians(angle)
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    
    def rotation_z(self, angle):
        """
        angle on z axis to rotation matrix
        """
        angle = np.radians(angle)
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    
    def rotation_xyz(self, x, y, z):
        """
        camera angle in world on x,y,z axis as rotation matrix
        """
        R = np.dot(self.rotation_z(z), np.dot(self.rotation_y(y), self.rotation_x(x)))
        return R

    def handle_camera(self, top_view=False, position=(0, 0, 0)):
        """
        handle keyboard input, update camera position and prepare for rendering
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.freeze = not self.freeze
                self.key_update(event.key)
        R = self.rotation_xyz(self.x_rot, self.y_rot, self.z_rot)
        t = np.array([self.translate_x, self.translate_y, 0.0])
        stack_r = np.vstack((R, [0, 0, 0]))
        stack_t = np.hstack((t.T, [1]))
        transformation_matrix = np.column_stack((stack_r, stack_t))
        if top_view == True:
            self.top_view(position)
            return
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(*position)
        glMultMatrixf(transformation_matrix.T)
        glScalef(self.scale_factor, self.scale_factor, self.scale_factor)
    
    def draw_points(self, points, color=(0.0, 1.0, 0.0)):
        glBegin(GL_POINTS)
        for point in points:
            glColor3f(color[0], color[1], color[2])
            glVertex3f(point[0], point[1], point[2])
        glEnd()

    def draw_lines(self, start, end, color=(0.0, 1.0, 1.0)):
        glBegin(GL_LINES)
        glColor3f(color[0], color[1], color[2])
        glVertex3f(start[0], start[1], start[2])
        glVertex3f(end[0], end[1], end[2])
        glEnd()

    def render3dSpace(self, points, position):
        if points is None:
            return
        print("pos", position)
        for point in points:
            point[0] = point[0] + position[0]
            point[1] = point[1] + position[1]
            point[2] = point[2] + position[2]
            self.points_stack.append(point)
        self.draw_points(self.points_stack)
        self.draw_lines([0, 0, 0], position)

    def render(self, milliseconds=10):
        pygame.display.flip()
        pygame.time.wait(milliseconds)

    def top_view(self, position):
        glLoadIdentity()
        R = np.array([[1, 0, 0],
                      [0, 0, 1],
                      [0, -1, 0]]) 
        t = np.array([0, 0, 0])
        R_full = np.vstack((R, [0, 0, 0]))
        position = np.array(position)
        position = np.hstack((position, [1]))
        transformation_matrix = np.column_stack((R_full, position))
        glMultMatrixf(transformation_matrix.T)
        glTranslatef(t[0], t[1], t[2])

    def draw_axes(self):
        #glTranslatef(0.0, 0.0, 0.0)
        glBegin(GL_LINES)
        # draw line for x axis
        unit = 100.0
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(-unit, 0.0, 0.0)
        glVertex3f(unit, 0.0, 0.0)
        # draw line for y axis
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, unit, 0.0)
        glVertex3f(0.0, -unit, 0.0)
        # draw line for Z axis
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, unit)
        glVertex3f(0.0, 0.0, -unit)
        glEnd()
        glutSwapBuffers()

    def draw_cube(self, position, scale=1):
        glScalef(scale, scale, scale)
        vertices = (
            (1, -1, -1), (1, 1, -1), (-1, 1, -1),
            (-1, -1, -1), (1, -1, 1), (1, 1, 1),
            (-1, -1, 1), (-1, 1, 1)
        )
        edges = (
            (0, 1),
            (0, 3),
            (0, 4),
            (2, 1),
            (2, 3),
            (2, 7),
            (6, 3),
            (6, 4),
            (6, 7),
            (5, 1),
            (5, 4),
            (5, 7)
        )
        faces = [
            (0, 1, 2, 3),
            (3, 2, 7, 6),
            (6, 7, 5, 4),
            (4, 5, 1, 0),
            (1, 5, 7, 2),
            (4, 0, 3, 6)
        ]
        glBegin(GL_QUADS)
        for face in faces:
            for vertex in face:
                glColor3fv((1,0,0))  # Any color you like
                glVertex3fv(vertices[vertex])
        glEnd()

        glColor3fv((0,0,0))
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()


    def draw_plane(self):
        #glTranslatef(0.0, -0.5, 0.0)
        # Define the vertices for the plane (adjust size and position as needed)
        plane_vertices = [
            (-100, 0.0, -100),
            (100, 0.0, -100),
            (100, 0.0, 100),
            (-100, 0.0, 100)
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
