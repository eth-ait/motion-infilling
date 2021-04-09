import time

import numpy as np
import pyglet
from tbase import utils
from pyglet.gl import *
from tbase.skeleton import Skeleton

RNG = np.random.RandomState(42)


def get_time_millis():
    """Returns current system time in milliseconds"""
    return int(round(time.time()*1000))


def setup_ortho(width, height):
    """Set up an orthogonal projection, which can be used to draw something in image coordinates."""
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, width, 0, height)


class LegendEntry(object):
    def __init__(self, text, color, font_size=10):
        self.text = text + ' --'
        self.color = color
        self.font_size = font_size

    def to_pyglet_label(self):
        return pyglet.text.Label(self.text, color=self.color, font_size=self.font_size,
                                 anchor_x='right', anchor_y='top')


class Legend(object):
    """A class to help drawing legends composed of labels."""
    def __init__(self, entries):
        self._labels = [e.to_pyglet_label() for e in entries]

    @classmethod
    def from_single_entry(cls, entry):
        return cls([entry])

    @classmethod
    def from_list(cls, entries):
        return cls(entries)

    @classmethod
    def empty(cls):
        return cls([])

    def is_empty(self):
        return len(self._labels) == 0

    def add_entry(self, entry):
        self._labels.append(entry.to_pyglet_label())

    def add_entries(self, entries):
        self._labels + [e.to_pyglet_label() for e in entries]

    def clear_all(self):
        self._labels = []

    def draw(self, x, y, height_per_label):
        for i in range(len(self._labels)):
            l = self._labels[i]
            l.x = x
            l.y = y - i*height_per_label
            l.draw()


# noinspection PyUnusedLocal
class Visualizer(pyglet.window.Window):
    """
    A slim motion capture viewer.
    """
    def __init__(self, caption='', show_floor=True, draw_cylinders=True):
        super(Visualizer, self).__init__(width=1100, height=800, resizable=True, caption=caption)

        self.zoom = 12
        self.ty = -3
        self.tz = -0.5
        # self.zoom = 25
        # self.ty = 0
        # self.tz = -1
        self.ry = 19
        self.rz = 51
        self.scale = 10.0

        # create the geometry for the floor with a checkerboard pattern
        black = [100, 100, 100] * 6
        white = [150, 150, 150] * 6
        n = 50
        z = 0
        vtx = []
        for i in range(n, -n, -1):
            for j in range(-n, n, 1):
                # two triangles per checkerboard tile
                vtx.extend((j, i, z,
                            j, i-1, z,
                            j+1, i, z,
                            j+1, i, z,
                            j, i-1, z,
                            j+1, i-1, z))

        self.floor = pyglet.graphics.vertex_list(
            len(vtx) // 3,
            ('v3f/static', vtx),
            ('c3B/static', ((black + white)*n + (white + black)*n)*n),
            ('n3i/static', [0, 0, 1] * (len(vtx) // 3)))
        self.coordinate_axes = pyglet.graphics.vertex_list(
            6,
            ('v3f/static', [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]),
            ('c3B/static', [0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 255, 50, 255, 255, 50]))

        self._frame_rate = None
        self._show_floor = show_floor
        self._paused = False
        self._show_foot_contacts = False
        self._chosen_foot = None
        self._winding = False
        self._background_white = False
        self._skeleton_sequences = []
        self._now_infilling = False
        self._draw_cylinders = draw_cylinders

        self._paused_label = pyglet.text.Label(
            'PAUSED',
            font_size=25,
            color=(255, 255, 0, 255),
            anchor_x='left', anchor_y='top')
        self._infilling_label = pyglet.text.Label(
            'NOW INFILLING',
            font_size=25,
            color=(0, 255, 0, 255),
            anchor_x='left', anchor_y='top')
        self._frame_label = pyglet.text.Label(
            '', x=120, y=10,
            font_size=24, bold=True,
            color=(127, 127, 127, 127))

        self._legend = Legend.empty()
        self._fps_display = pyglet.window.FPSDisplay(self)

        self.set_location(50, 50)
        self.setup_opengl()

    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)

        light0pos = [20.0, 20.0, 20.0, 1.0]
        glLightfv(GL_LIGHT0, GL_POSITION, utils.vec(*light0pos))
        glLightfv(GL_LIGHT0, GL_AMBIENT, utils.vec(0.3, 0.3, 0.3, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, utils.vec(0.9, 0.9, 0.9, 1.0))
        glLightfv(GL_LIGHT0, GL_SPECULAR, utils.vec(1.0, 1.0, 1.0, 1.0))

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, utils.vec(0.8, 0.5, 0.5, 1.0))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, utils.vec(1, 1, 1, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)

    def _display_loaded_animation(self):
        if self._skeleton_sequences is not None and len(self._skeleton_sequences) > 0:
            [s.reset_time() for s in self._skeleton_sequences]
            pyglet.clock.schedule_interval(self._update, 1.0 / 60.0)

    def _update(self, dt):
        if self._paused:
            return
        [s.fast_forward() for s in self._skeleton_sequences]

    def _set_background_color(self):
        if self._background_white:
            glClearColor(1.0, 1.0, 1.0, 0.0)
        else:
            glClearColor(0.0, 0.0, 0.0, 0.0)

    def _get_current_frame_nr(self):
        if self._skeleton_sequences is not None and len(self._skeleton_sequences) > 0:
            return self._skeleton_sequences[0].frame_pointer
        else:
            return -1

    def _draw_2d_elements(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        setup_ortho(self.width, self.height)

        if self._paused:
            self._paused_label.x = 0
            self._paused_label.y = self.height
            self._paused_label.draw()

        if self._now_infilling:
            self._infilling_label.x = self.width/2-20
            self._infilling_label.y = self.height
            self._infilling_label.draw()

        if not self._legend.is_empty():
            self._legend.draw(self.width, self.height - 30, 15)

        self._fps_display.draw()

        self._frame_label.text = '#{}'.format(self._get_current_frame_nr())
        self._frame_label.draw()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def load_skeleton_sequences(self, sequence):
        assert isinstance(sequence, list), 'expecting a list of sequences'
        self._skeleton_sequences = sequence
        # construct legend
        for sk in sequence:
            color = [int(round(c * 255)) for c in sk.color]
            self._legend.add_entry(LegendEntry(sk.name, color))
        self._display_loaded_animation()

    def clear_skeleton_sequences(self):
        pyglet.clock.unschedule(self._update)
        self._skeleton_sequences = []
        self._legend = Legend.empty()

    def show_(self):
        pyglet.app.run()

    def enable_show_foot_contacts(self, value):
        self._show_foot_contacts = value

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glu.gluPerspective(45, float(width) / height, 1, 100)

    def on_mouse_scroll(self, x, y, dx, dy):
        if dy == 0:
            return
        self.zoom *= 1.1 ** (-1 if dy < 0 else 1)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifier):
        if self._winding and self._paused:
            # rewind if mouse is moving left
            if dx < 0:
                [s.rewind() for s in self._skeleton_sequences]
            elif dx > 0:
                [s.fast_forward() for s in self._skeleton_sequences]
        else:
            if buttons == pyglet.window.mouse.RIGHT:
                self.ty += 0.05 * dx
                self.tz += 0.05 * dy
            else:
                self.ry += 0.2 * -dy
                self.rz += 0.2 * dx

    def on_mouse_press(self, x, y, buttons, modifier):
        pass

    def on_mouse_release(self, x, y, button, modifiers):
        pass

    def on_mouse_motion(self, x, y, buttons, modifier):
        pass

    # noinspection PyProtectedMember
    def on_key_press(self, key, modifier):
        k = pyglet.window.key
        if key == k.ESCAPE:
            # exit the application
            pyglet.app.exit()
        elif key == k.SPACE:
            # pause the animation
            self._paused = not self._paused
        elif key == k.LEFT:
            # revert one time step if paused
            if self._paused:
                [s.rewind() for s in self._skeleton_sequences]
        elif key == k.RIGHT:
            # advance on time step if paused
            if self._paused:
                [s.fast_forward() for s in self._skeleton_sequences]
        elif key == k.H:
            # toggle if floor should be hidden or not
            self._show_floor = not self._show_floor
        elif key == k.B:
            # toggle white or black color for background
            self._background_white = not self._background_white
        elif key == k.F:
            # toggle drawing of foot contact information
            self.enable_show_foot_contacts(not self._show_foot_contacts)
        elif key == k.S:
            # export all clips to CSV
            [sk.export_to_csv() for sk in self._skeleton_sequences]
        elif key == k.NUM_1 or key == k._1:
            self._chosen_foot = Skeleton.idx_to_foot(0)
        elif key == k.NUM_2 or key == k._2:
            self._chosen_foot = Skeleton.idx_to_foot(1)
        elif key == k.NUM_3 or key == k._3:
            self._chosen_foot = Skeleton.idx_to_foot(2)
        elif key == k.NUM_4 or key == k._4:
            self._chosen_foot = Skeleton.idx_to_foot(3)
        elif key == k.NUM_0 or key == k._0:
            self._chosen_foot = None
        elif key == k.W:
            self._winding = not self._winding

    def on_draw(self):
        self._set_background_color()
        self.clear()

        # material for anything which is not a skeleton
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, utils.vec(0.75, 0.75, 0.75, 1.0))

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1, 0, 0, 0, 0, 0, 0, 0, 1)
        glTranslatef(-self.zoom, 0, 0)
        glTranslatef(0, self.ty, self.tz)
        glRotatef(self.ry, 0, 1, 0)
        glRotatef(self.rz, 0, 0, 1)

        if self._show_floor:
            self.floor.draw(GL_TRIANGLES)
        self.coordinate_axes.draw(GL_LINES)

        glLineWidth(2.0)
        for i, seq in enumerate(self._skeleton_sequences):
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glScalef(1.0 / self.scale, 1.0 / self.scale, 1.0 / self.scale)

            glPointSize(2.0)
            seq.draw_root_trajectory(full=False)

            if self._show_foot_contacts:
                glLineWidth(1.0)
                seq.draw_foot_contacts(self._chosen_foot)
                glLineWidth(2.0)

            self._now_infilling = seq.is_infilling()

            glPointSize(3.0)
            seq.draw_current_timestep(draw_cylinders=self._draw_cylinders)
            seq.draw_static_frames(draw_cylinders=self._draw_cylinders)
            seq.draw_static_points()

            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
        glLineWidth(1.0)

        self._draw_2d_elements()
