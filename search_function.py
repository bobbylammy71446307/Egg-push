import numpy as np
from copy import deepcopy
import random as rand
from scipy.optimize import linear_sum_assignment
from queue import PriorityQueue

directions={"up":[-1,0],
            "down":[1,0],
            "left":[0,-1],
            "right":[0,1]}

class Map():

    def __init__(self,row=10,column=12,block_num=8):
        self.row=row
        self.column=column
        self.block_num=block_num
        self.cost_matrix=None
        self.create_map()

    def create_map(self):
        map_valid=False
        while map_valid==False:
            self.map=np.zeros((self.row,self.column))
            self.init_wall()
            self.init_egg()
            self.init_hole()
            map_valid=self.validate_map()
    
    def init_wall(self):
        # Create walls around the map
        self.map[0, :] = 1          # Top wall
        self.map[-1, :] = 1         # Bottom wall
        self.map[:, 0] = 1          # Left wall
        self.map[:, -1] = 1

        #Create mouse
        self.map[self.row-2][1]=2

        # Create random walls inside the map
        block_num=0
        while block_num<self.block_num:
            wall=[rand.randint(1,self.row-2),rand.randint(1,self.column-2)]
            if self.map[wall[0]][wall[1]]==0:
                self.map[wall[0]][wall[1]]=1
                block_num+=1
    
    def init_egg(self):
        self.egg_list=np.empty((0,2),dtype=int)
        while len(self.egg_list)<4:
            egg=[rand.randint(2,self.row-3),rand.randint(2,self.column-3)]
            if self.map[egg[0]][egg[1]]==0:
                self.map[egg[0]][egg[1]]=3
                self.egg_list = np.vstack([self.egg_list, egg])  
    
    def init_hole(self):
        self.hole_list=np.empty((0,2),dtype=int)
        while len(self.hole_list)<4:
            hole=[rand.randint(1,self.row-2),rand.randint(1,self.column-2)]
            if self.map[hole[0]][hole[1]]==0:
                self.map[hole[0]][hole[1]]=4
                self.hole_list = np.vstack([self.hole_list, hole])  

    def validate_map(self):
        if self.map[7][1]==1 or self.map[8][2]==1:
            #print("Invalid map: Mouse is surrounded by walls")
            return False

        for i in range(4):
            egg_coord = self.egg_list[i]
            hole_coord = self.hole_list[i]
            
            # Check egg is not surrounded by too many walls
            egg_wall_count = 0
            hole_wall_count = 0
            
            for direction in directions.values():
                dx, dy = direction
                x, y = egg_coord[0] + dx, egg_coord[1] + dy
                # Check boundaries
                if 0 <= x < self.row and 0 <= y < self.column:
                    if self.map[x, y] == 1:
                        egg_wall_count += 1
                
                x, y = hole_coord[0] + dx, hole_coord[1] + dy
                if 0 <= x < self.row and 0 <= y < self.column:
                    if self.map[x, y] == 1:
                        hole_wall_count += 1
                if egg_wall_count > 1:
                    #print(f"Egg at {egg_coord} is surrounded by too many walls")
                    return False
                
                if hole_wall_count == 4:
                    #print(f"Hole at {hole_coord} is completely surrounded by walls")
                    return False
        #print("Map is valid")
        return True

    def assign_eggs_to_holes(self):
        self.cost_matrix = np.zeros((len(self.egg_list), len(self.hole_list)))
        # Calculate the cost matrix based on Manhattan distance
        for i, egg in enumerate(self.egg_list):
            for j,hole in enumerate(self.hole_list):
                egg_hole_distance=abs(egg[0]-hole[0])+abs(egg[1]-hole[1])
                self.cost_matrix[i][j]=egg_hole_distance
        # Solve the assignment problem using the Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(self.cost_matrix)
        assignments = {egg_idx: hole_idx for egg_idx, hole_idx in zip(row_indices, col_indices)}        
        return assignments

    def greedy_mouse_visit(self,assignment_dict):
        unvisited_eggs=self.egg_list.copy()
        current_pos=[self.row-2,1]
        mouse_visit=[]
        while len(unvisited_eggs)>0:
            min_dist=self.column+self.row
            for idx,egg in enumerate(unvisited_eggs):
                distance=abs(egg[0]-current_pos[0])+abs(egg[1]-current_pos[1])
                if distance<min_dist:
                    min_dist=distance
                    nearest_egg=egg
                    nearest_idx=idx
            mouse_visit.append(nearest_egg)
            unvisited_eggs = np.delete(unvisited_eggs, nearest_idx, axis=0)
            egg_idx = np.where((self.egg_list == nearest_egg).all(axis=1))[0][0]
            current_pos=self.hole_list[assignment_dict[egg_idx]]
        return mouse_visit
        
class Node():
    def __init__(self,map,coordinate,goal,depth=0,direction=None,parent=None):
        self.map=map
        self.coordinate=coordinate
        self.goal=goal
        self.depth=depth
        self.direction=direction
        self.parent=parent
        self.cost=self.get_cost()
    
    def get_cost(self):
        heuristic_cost = abs(self.coordinate[0] - self.goal[0]) + abs(self.coordinate[1] - self.goal[1]) + self.get_surr_block()
        if self.parent is not None and self.direction!=self.parent.direction:
            heuristic_cost+=2
        
        return self.depth + heuristic_cost
    
    def get_surr_block(self):
        surr=[[1,0],[0,1],[-1,0],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
        block_cost=0
        for direction in surr:
            dx,dy=direction
            x,y=self.coordinate[0]+dx,self.coordinate[1]+dy
            if not (x==0 or x==self.map.shape[0] or y==0 or y==self.map.shape[1]):
                if self.map[x][y]!=0:
                    block_cost+=0.1
        return block_cost
                    
    def __lt__(self, other):
        return self.cost < other.cost  
    
    def __str__(self):
        return f"Node({self.coordinate}, {self.depth}, {self.cost})"
    
class A_star():
    def __init__(self,start,goal,map,is_mouse=False):
        self.is_mouse=is_mouse
        self.start=start
        self.goal=goal
        self.map=map
        self.open_list=PriorityQueue()
        self.closed_list=set()
        self.valid_cell=[0,4]

    def a_star(self):
        # Add the starting node to the open list
        start_node = Node(self.map,self.start,self.goal)
        self.open_list.put(start_node)
        #print("Start search from ",self.start,"to",self.goal)

        while self.open_list.qsize()> 0:
            # Sort the open list by f(n) = g(n) + h(n)
            current_node = self.open_list.get()

            if (current_node.coordinate == self.goal).all():
                #print("Found a path to the goal!")
                path=self.get_path(current_node)
                mouse_coordinate=self.get_mouse_position(path[1]) if not self.is_mouse else None
                return mouse_coordinate,path

            self.closed_list.add(tuple(current_node.coordinate))
            # Generate child nodes
            self.expand(current_node)
        #print("No path found to the goal.")
        return None,None


    def expand(self,node):
        for direction, coord in directions.items():
            dx, dy = coord
            new_coordinate = node.coordinate + np.array([dx, dy])
            mouse_coordinate=node.coordinate+np.array([-dx,-dy])
            if self.is_valid(new_coordinate,mouse_coordinate) and tuple(new_coordinate) not in self.closed_list:
                new_depth = node.depth + 1
                new_node = Node(self.map,new_coordinate,self.goal,new_depth,direction,node)
                self.open_list.put(new_node)

            
    def is_valid(self,coordinate,mouse_coordinate):
        x,y=coordinate
        mouse_x,mouse_y=mouse_coordinate
        if self.map[x][y] not in self.valid_cell:
            return False
        if x<0 or x>=self.map.shape[0] or y<0 or y>=self.map.shape[1]:
            return False
        if self.map[mouse_x][mouse_y]!=0 and not self.is_mouse:
            return False
        return True
    
    def get_path(self,node):
        path = []
        while node.parent is not None:
            path.append(node)
            node = node.parent
        if not self.is_mouse:
            path.append(node)
        return path[::-1]
    
    def get_mouse_position(self,first_node):
        mouse_pos = self.start-np.array(directions[first_node.direction])
        return mouse_pos
    
class Solver():
    def __init__(self):
        self.complete=False
        self.exit=False
        self.path=[]
        self.mouse_path=[]
    

    def init_map(self):
        self.m=Map()
        self.assignment=self.m.assign_eggs_to_holes()
        self.order=self.m.greedy_mouse_visit(self.assignment)
    
    def create_partial_map(self,egg_coordinate=None,hole_coordinate=None,partial_egg=None):
        partial_map=deepcopy(self.m.map)
        for i in range(len(partial_map)):
            for j in range(len(partial_map[i])):
                if any(np.array_equal(np.array([i,j]), coord) for coord in [egg_coordinate, hole_coordinate]):
                    partial_map[i][j]=0
                elif partial_map[i][j] in [3,4] or np.array_equal(np.array([i,j]),partial_egg):
                    partial_map[i][j]=1
                elif partial_map[i][j]==2:
                    partial_map[i][j]=0
        return partial_map


    def egg_hole_search(self,egg_coord):
        egg_index=np.where((self.m.egg_list == egg_coord).all(axis=1))[0][0]
        hole_index=self.assignment[egg_index]
        hole_coord=self.m.hole_list[hole_index]
        #print(self.create_partial_map(egg_coord,hole_coord))
        egg_hole_search=A_star(egg_coord,hole_coord,self.create_partial_map(egg_coord,hole_coord))
        return egg_hole_search.a_star()

    def mouse_hole_search(self,mouse_coordinate,mouse_nxt_to_egg):
        #print(self.create_partial_map())
        mouse_hole_search=A_star(mouse_coordinate,mouse_nxt_to_egg,self.create_partial_map(),is_mouse=True)
        return mouse_hole_search.a_star()

    def get_complete_path(self,mouse_path,path,egg_coordinate):
        last_is_dir=False
        compelete_path=[cell.coordinate for cell in mouse_path]
        last_direction=None
        for cell in path:
            if cell.direction == last_direction or last_direction is None:
                if cell.parent is not None:
                    last_coordinate=last_cell.coordinate
                    last_direction=cell.direction
                    compelete_path.append(last_coordinate)
                last_cell=cell
                last_is_dir=False
            else:
                if last_is_dir:
                    print("reentered")
                    last_coordinate=last_cell.parent.coordinate
                goal_pos=cell.parent.coordinate-np.array(directions[cell.direction])
                inter_search=A_star(last_coordinate,goal_pos,self.create_partial_map(egg_coordinate=egg_coordinate,partial_egg=cell.parent.coordinate),is_mouse=True)
                _,inter_path=inter_search.a_star()
                if inter_path is None:
                    return None
                for inter_cell in inter_path:
                    if inter_cell.parent is not None:
                        compelete_path.append(inter_cell.coordinate)
                compelete_path.append(last_cell.coordinate)
                last_coordinate=cell.coordinate
                last_direction=cell.direction
                last_cell=cell
                last_is_dir=True

        return compelete_path


 

    def main(self):
        while not self.complete:
            self.complete_path=[]
            self.exit=False
            self.init_map()
            print(self.m.map)
            mouse_coordinate=np.array([self.m.row-2,1])
            for visit_egg_coordinate in self.order:
                mouse_nxt_to_egg,path=self.egg_hole_search(visit_egg_coordinate)
                if path is None:
                    #print("No path found to the goal.")
                    self.exit=True
                    continue
                self.path.append(path)
                #print("path",[cell.coordinate for cell in path])
                #print("mouse next to egg",mouse_nxt_to_egg)
                
                _,mouse_path=self.mouse_hole_search(mouse_coordinate,mouse_nxt_to_egg)
                if mouse_path is None:
                    print("No path found to the goal.")
                    self.exit=True
                    continue
                #print("mouse path",[cell.coordinate for cell in mouse_path])

                complete_path=self.get_complete_path(mouse_path,path,visit_egg_coordinate)
                if complete_path is None:
                    print("No path found to the goal.")
                    self.exit=True
                    continue

                self.complete_path+=complete_path


                
                self.m.map[mouse_coordinate[0]][mouse_coordinate[1]]=0
                mouse_coordinate=path[-2].coordinate
                self.m.map[mouse_coordinate[0]][mouse_coordinate[1]]=2
                self.m.map[visit_egg_coordinate[0]][visit_egg_coordinate[1]]=0
                
            if not self.exit:
                #print("search complete")
                print(self.complete_path)
                self.complete=True
        return self.complete_path





if __name__ == "__main__":
    S=Solver()
    S.main()
