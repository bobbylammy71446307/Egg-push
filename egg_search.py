import copy
import numpy as np
import pygame
from search_function import Map, Solver, SokobanSolver

FPS = 5
WHITE = (255, 255, 255)
GREY = (169, 169, 169)
DARK = (40, 40, 40)
BLUE = (60, 110, 200)
GREEN = (50, 160, 90)
RED = (200, 70, 70)

CELL = 60
COLS = 12
ROWS = 10
PANEL_W = 280
GRID_W = COLS * CELL
GRID_H = ROWS * CELL
HEADER_H = 60
WIDTH = GRID_W * 2 + PANEL_W
HEIGHT = HEADER_H + GRID_H + 40


class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, ox, oy):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((CELL, CELL))
        self.image.fill(GREY)
        self.rect = self.image.get_rect()
        self.rect.x = ox + x * CELL
        self.rect.y = oy + y * CELL


class Egg(pygame.sprite.Sprite):
    def __init__(self, x, y, ox, oy):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('image/egg.png')
        self.image = pygame.transform.scale(self.image, (CELL, CELL))
        self.rect = self.image.get_rect()
        self.ox, self.oy = ox, oy
        self.rect.x = ox + x * CELL
        self.rect.y = oy + y * CELL


class Mouse(pygame.sprite.Sprite):
    def __init__(self, x, y, ox, oy):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('image/mouse.jpeg')
        self.image = pygame.transform.scale(self.image, (CELL, CELL))
        self.rect = self.image.get_rect()
        self.ox, self.oy = ox, oy
        self.rect.x = ox + x * CELL
        self.rect.y = oy + y * CELL


class Hole(pygame.sprite.Sprite):
    def __init__(self, x, y, ox, oy):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('image/hole.png')
        self.image = pygame.transform.scale(self.image, (CELL, CELL))
        self.rect = self.image.get_rect()
        self.rect.x = ox + x * CELL
        self.rect.y = oy + y * CELL


class Complete(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('image/hole_complete.png')
        self.image = pygame.transform.scale(self.image, (CELL, CELL))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y


class GridView:
    """One of the two side-by-side panels. Owns its sprites and steps its own trajectory."""

    def __init__(self, title, solver, trajectory, origin):
        self.title = title
        self.solver = solver
        self.trajectory = trajectory
        self.origin = origin
        self.idx = 0
        self.prev_mouse_coord = None
        self.finished = False

        self.all_sprites = pygame.sprite.Group()
        self.eggs = pygame.sprite.Group()
        self.holes = pygame.sprite.Group()
        self.completed = pygame.sprite.Group()
        self.mouse_group = pygame.sprite.Group()
        self.mouse_sprite = None
        self._build_sprites()

    def _build_sprites(self):
        ox, oy = self.origin
        m = self.solver.m
        for i in range(m.row):
            for j in range(m.column):
                v = m.map[i][j]
                if v == 1:
                    self.all_sprites.add(Wall(j, i, ox, oy))
                elif v == 3:
                    e = Egg(j, i, ox, oy)
                    self.all_sprites.add(e)
                    self.eggs.add(e)
                elif v == 4:
                    h = Hole(j, i, ox, oy)
                    self.all_sprites.add(h)
                    self.holes.add(h)
        ms = Mouse(1, m.row - 2, ox, oy)
        self.all_sprites.add(ms)
        self.mouse_group.add(ms)
        self.mouse_sprite = ms

    def step(self):
        if self.idx >= len(self.trajectory):
            self.finished = True
            return
        ox, oy = self.origin
        move = self.trajectory[self.idx]
        self.mouse_sprite.rect.x = ox + move[1] * CELL
        self.mouse_sprite.rect.y = oy + move[0] * CELL
        self.prev_mouse_coord = move if self.idx == 0 else self.trajectory[self.idx - 1]
        self.idx += 1
        self._handle_collisions()

    def _handle_collisions(self):
        ox, oy = self.origin
        eh = pygame.sprite.groupcollide(self.eggs, self.holes, False, True)
        for egg in eh:
            c = Complete(egg.rect.x, egg.rect.y)
            self.all_sprites.add(c)
            self.completed.add(c)
            egg.kill()
        me = pygame.sprite.groupcollide(self.mouse_group, self.eggs, False, False)
        for _, eggs_hit in me.items():
            for egg in eggs_hit:
                grid = np.array([(egg.rect.y - oy) // CELL, (egg.rect.x - ox) // CELL])
                new = grid + (grid - np.array(self.prev_mouse_coord))
                egg.rect.x = ox + new[1] * CELL
                egg.rect.y = oy + new[0] * CELL

    def done(self):
        return len(self.completed) == 4 or self.idx >= len(self.trajectory)


class SideBySide:
    def __init__(self, shared_map):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Egg Search — Legacy vs Sokoban")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.font_big = pygame.font.SysFont("monospace", 22, bold=True)

        legacy = Solver()
        legacy_path = legacy.main(shared_map=copy.deepcopy(shared_map))
        legacy_traj = [np.array(p) for p in legacy_path]

        sokoban = SokobanSolver()
        sokoban_path = sokoban.main(shared_map=copy.deepcopy(shared_map))
        sokoban_traj = [np.array(p) for p in sokoban_path]

        self.left = GridView("Legacy (decomposed A*)", legacy,
                             legacy_traj, origin=(0, HEADER_H))
        self.right = GridView("Sokoban (unified A*)", sokoban,
                              sokoban_traj, origin=(GRID_W + PANEL_W, HEADER_H))

    def _draw_header(self):
        pygame.draw.rect(self.screen, DARK, (0, 0, WIDTH, HEADER_H))
        left_title = self.font_big.render(self.left.title, True, BLUE)
        right_title = self.font_big.render(self.right.title, True, GREEN)
        self.screen.blit(left_title, (20, 18))
        self.screen.blit(right_title, (GRID_W + PANEL_W + 20, 18))

    def _draw_panel(self):
        x0 = GRID_W
        pygame.draw.rect(self.screen, (245, 245, 245), (x0, HEADER_H, PANEL_W, GRID_H))
        pygame.draw.line(self.screen, DARK, (x0, HEADER_H), (x0, HEADER_H + GRID_H), 2)
        pygame.draw.line(self.screen, DARK, (x0 + PANEL_W, HEADER_H),
                         (x0 + PANEL_W, HEADER_H + GRID_H), 2)

        def row(y, label, lv, rv, lc=BLUE, rc=GREEN):
            self.screen.blit(self.font.render(label, True, DARK), (x0 + 10, y))
            self.screen.blit(self.font.render(lv, True, lc), (x0 + 10, y + 18))
            self.screen.blit(self.font.render(rv, True, rc), (x0 + 10, y + 36))

        ls, rs = self.left.solver.stats, self.right.solver.stats
        y = HEADER_H + 20
        row(y, "compute time", f"L  {ls['elapsed']*1000:8.1f} ms",
            f"R  {rs['elapsed']*1000:8.1f} ms"); y += 70
        row(y, "path length (steps)", f"L  {ls['steps']}",
            f"R  {rs['steps']}"); y += 70
        row(y, "nodes expanded", f"L  {ls['nodes_expanded']}",
            f"R  {rs['nodes_expanded']}"); y += 70
        row(y, "A* calls", f"L  {ls['astar_calls']}",
            f"R  {rs['astar_calls']}"); y += 70
        row(y, "map retries", f"L  {ls['retries']}",
            f"R  {rs['retries']}"); y += 70

        progress = self.font.render(
            f"step  L {self.left.idx}/{len(self.left.trajectory)}   "
            f"R {self.right.idx}/{len(self.right.trajectory)}",
            True, DARK)
        self.screen.blit(progress, (x0 + 10, HEADER_H + GRID_H - 30))

    def draw(self):
        self.screen.fill(WHITE)
        self._draw_header()
        self.left.all_sprites.draw(self.screen)
        self.right.all_sprites.draw(self.screen)
        self._draw_panel()
        pygame.display.update()

    def main(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not self.left.finished:
                self.left.step()
            if not self.right.finished:
                self.right.step()
            self.draw()
            if self.left.finished and self.right.finished:
                pygame.time.wait(2000)
                running = False


if __name__ == "__main__":
    shared = Map(row=ROWS, column=COLS)
    viz = SideBySide(shared)
    viz.main()
    pygame.quit()
