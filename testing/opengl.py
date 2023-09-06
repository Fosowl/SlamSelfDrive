import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import * 


def draw_lines():
    glBegin(GL_LINES)

    glColor3f(1.0, 0.0, 0.0)  # Red color
    glVertex3f(0.0, 0.0, 0.0)  # Start point
    glVertex3f(1.0, 1.0, 0.0)  # End point

    glColor3f(1.0, 0.0, 0.0)  # Red color
    glVertex3f(0.0, 0.0, 1.0)  # Start point
    glVertex3f(1.0, 1.0, 1.0)  # End point

    glEnd()

def draw_triangle():
    glBegin(GL_TRIANGLES)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-1.0, -1.0, 0.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(1.0, -1.0, 0.0)
    glEnd()

def main():
    x_rot = 0
    y_rot = 0
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    rotate_speed_x = 0.0
    rotate_speed_y = 0.0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    rotate_speed_y += 0.1  # X-axis rotation
                elif event.key == pygame.K_k:
                    rotate_speed_y -= 0.1  # X-axis rotation
                if event.key == pygame.K_j:
                    rotate_speed_x -= 0.1  # Y-axis rotation
                elif event.key == pygame.K_l:
                    rotate_speed_x += 0.1
        x_rot += rotate_speed_y
        y_rot += rotate_speed_x
        print("rotation: ", x_rot, y_rot)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, 0.0)
        glRotatef(x_rot, 1, 0, 0)  # Rotate around X-axis
        glRotatef(y_rot, 0, 1, 0)  # Rotate around Y-axis
        #draw_lines()
        draw_triangle()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    main()