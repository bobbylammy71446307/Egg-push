import numpy as np
import pygame
from search_function import Solver


FPS=5
WHITE=(255,255,255)
GREY=(169,169,169)
WIDTH=1200
HEIGHT=1000

class Wall(pygame.sprite.Sprite):

    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.Surface((100,100))
        self.image.fill(GREY)
        self.rect=self.image.get_rect()
        self.rect.x=x*100
        self.rect.y=y*100


class Egg(pygame.sprite.Sprite):
    def __init__(self,x,y,ori_pos=[0,0]):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.image.load('image/egg.png')
        self.image=pygame.transform.scale(self.image,(100,100))
        self.rect=self.image.get_rect()
        self.rect.x=x*100
        self.rect.y=y*100
        self.ori_pos=ori_pos
    
    def update(self):

        if self.rect.right>WIDTH-100:
            self.rect.right=WIDTH-100
        if self.rect.left<100:
            self.rect.left=100
        if self.rect.top<100:
            self.rect.top=100
        if self.rect.bottom>HEIGHT-100:
            self.rect.bottom=HEIGHT-100


class Mouse(pygame.sprite.Sprite):
    def __init__(self,x,y,ori_pos=[0,0]):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.image.load('image/mouse.jpeg')
        self.image=pygame.transform.scale(self.image,(100,100))
        self.rect=self.image.get_rect()
        self.rect.x=x*100
        self.rect.y=y*100
        self.ori_pos=ori_pos
    def update(self):
        if self.rect.right>WIDTH-100:
            self.rect.right=WIDTH-100
        if self.rect.left<100:
            self.rect.left=100
        if self.rect.top<100:
            self.rect.top=100
        if self.rect.bottom>HEIGHT-100:
            self.rect.bottom=HEIGHT-100

class Hole(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.image.load('image/hole.png')
        self.image=pygame.transform.scale(self.image,(100,100))
        self.rect=self.image.get_rect()
        self.rect.x=x*100
        self.rect.y=y*100


class Complete(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)
        self.image=pygame.image.load('image/hole_complete.png')
        self.image=pygame.transform.scale(self.image,(100,100))
        self.rect=self.image.get_rect()
        self.rect.x=x
        self.rect.y=y


class Finish(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self) 
        self.image=pygame.image.load('image/finish.jpeg')
        self.image=pygame.transform.scale(self.image,(500,500))
        self.rect=self.image.get_rect()
        self.rect.x=WIDTH/4
        self.rect.y=HEIGHT/4

class Pygame_Vsiualization:
    def __init__(self,solver,trajectory):
        self.solver=solver
        self.trajectory=trajectory
        self.trajectory_index=0
        pygame.init()
        self.screen=pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("Egg Search")    
        self.clock=pygame.time.Clock()
        self.all_sprites=pygame.sprite.Group()
        self.blocks=pygame.sprite.Group()
        self.eggs=pygame.sprite.Group()
        self.holes=pygame.sprite.Group()
        self.completed=pygame.sprite.Group()
        self.mouse=pygame.sprite.Group()
        self.mouse_sprite=None
        self.prev_mouse_coord=None
        


    def add_sprite(self):
        for i in range (self.solver.m.row):
            for j in range (self.solver.m.column):
                if self.solver.m.map[i][j]==1:
                    sprite=Wall(j,i)
                elif self.solver.m.map[i][j]==3:
                    sprite=Egg(j,i)
                elif self.solver.m.map[i][j]==4:
                    sprite=Hole(j,i)
                else:
                    continue
                self.add(sprite)
        for egg_coord in self.solver.order:
            sprite=Egg(egg_coord[1],egg_coord[0])
            self.add(sprite)
        self.add(Mouse(1,self.solver.m.row-2))

    def add(self,sprite):
        self.all_sprites.add(sprite)
        if isinstance(sprite, Wall):
            self.blocks.add(sprite)
        elif isinstance(sprite, Egg):
            self.eggs.add(sprite)
        elif isinstance(sprite, Hole):
            self.holes.add(sprite)
        elif isinstance(sprite, Complete):
            self.completed.add(sprite)
        elif isinstance(sprite, Mouse):
            self.mouse.add(sprite)
            self.mouse_sprite=sprite

    def update(self):
        self.all_sprites.update()

    def draw(self):
        self.screen.fill(WHITE)    
        self.all_sprites.draw(self.screen)
        pygame.display.update()

    def play_trajectory_step(self):
        if self.trajectory_index<len(self.trajectory):
            move=self.trajectory[self.trajectory_index]
            self.mouse_sprite.rect.x=move[1]*100
            self.mouse_sprite.rect.y=move[0]*100
            self.prev_mouse_coord=move if self.trajectory_index==0 else self.trajectory[self.trajectory_index-1]
            self.trajectory_index+=1
            
    def update_egg_position(self,egg):
        egg_coordinate=np.array([egg.rect.y//100,egg.rect.x//100])
        egg_new_coordinate=egg_coordinate + (egg_coordinate-self.prev_mouse_coord)
        egg.rect.x=egg_new_coordinate[1]*100
        egg.rect.y=egg_new_coordinate[0]*100

    def collide(self):
        egg_hole_collide=pygame.sprite.groupcollide(self.eggs,self.holes,False,True)
        mouse_egg_collide=pygame.sprite.groupcollide(self.mouse,self.eggs,False,False)
        for hit in egg_hole_collide:
            for entity in self.eggs:
                if entity in egg_hole_collide:
                    collide_egg=entity
                    complete=Complete(collide_egg.rect.x,collide_egg.rect.y)
                    self.all_sprites.add(complete)
                    self.completed.add(complete)
        for push in mouse_egg_collide:
            for egg in mouse_egg_collide[push]:
                self.update_egg_position(egg)


    def main(self):
        self.add_sprite()
        running=True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running=False
            self.play_trajectory_step()
            self.collide()
            if len(self.completed)==4:
                finish=Finish()
                self.all_sprites.add(finish)
                running=False
            self.update()
            self.draw()



if __name__ =="__main__":
    S=Solver()
    mouse_trajectory=S.main()
    viz=Pygame_Vsiualization(S,mouse_trajectory)
    viz.main()
    pygame.quit()

