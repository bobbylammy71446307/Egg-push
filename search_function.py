import numpy as np
from copy import deepcopy
import random as rand
from scipy.optimize import linear_sum_assignment
from queue import PriorityQueue
import heapq
import time
from collections import deque

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
        self.nodes_expanded=0

    def a_star(self):
        # Add the starting node to the open list
        start_node = Node(self.map,self.start,self.goal)
        self.open_list.put(start_node)
        #print("Start search from ",self.start,"to",self.goal)

        while self.open_list.qsize()> 0:
            # Sort the open list by f(n) = g(n) + h(n)
            current_node = self.open_list.get()
            self.nodes_expanded+=1

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
        self.stats={"nodes_expanded":0,"astar_calls":0,"retries":0,"elapsed":0.0,"steps":0}
    

    def init_map(self):
        self.m=Map()
        self.assignment=self.m.assign_eggs_to_holes()
        self.order=self.m.greedy_mouse_visit(self.assignment)

    def load_map(self,shared_map):
        self.m=shared_map
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
        result=egg_hole_search.a_star()
        self.stats["astar_calls"]+=1
        self.stats["nodes_expanded"]+=egg_hole_search.nodes_expanded
        return result

    def mouse_hole_search(self,mouse_coordinate,mouse_nxt_to_egg):
        #print(self.create_partial_map())
        mouse_hole_search=A_star(mouse_coordinate,mouse_nxt_to_egg,self.create_partial_map(),is_mouse=True)
        result=mouse_hole_search.a_star()
        self.stats["astar_calls"]+=1
        self.stats["nodes_expanded"]+=mouse_hole_search.nodes_expanded
        return result

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
                self.stats["astar_calls"]+=1
                self.stats["nodes_expanded"]+=inter_search.nodes_expanded
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


 

    def main(self,shared_map=None):
        t0=time.perf_counter()
        while not self.complete:
            self.complete_path=[]
            self.exit=False
            if shared_map is not None:
                self.load_map(shared_map)
                shared_map=None
            else:
                self.init_map()
                self.stats["retries"]+=1
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
        self.stats["elapsed"]=time.perf_counter()-t0
        self.stats["steps"]=len(self.complete_path)
        return self.complete_path





class SokobanSolver():
    """Unified state-space A*: one search over (mouse_pos, frozenset(egg_positions)).

    Produces a globally optimal push plan. The trajectory returned matches the
    legacy Solver: a list of [row, col] mouse positions, one per tick, where
    a push is a tick in which the mouse steps onto an egg's cell and the egg
    advances by one cell in the same direction.
    """

    def __init__(self):
        self.stats={"nodes_expanded":0,"astar_calls":1,"retries":0,"elapsed":0.0,"steps":0}
        self.complete_path=[]
        self.m=None

    def init_map(self):
        self.m=Map()

    def load_map(self,shared_map):
        self.m=shared_map

    def _passable(self,r,c):
        if r<0 or r>=self.m.row or c<0 or c>=self.m.column:
            return False
        return self.m.map[r][c]!=1

    def _bfs_distance_map(self,target):
        INF=10**9
        dist=[[INF]*self.m.column for _ in range(self.m.row)]
        dist[target[0]][target[1]]=0
        q=deque([(target[0],target[1])])
        while q:
            r,c=q.popleft()
            for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr,nc=r+dr,c+dc
                if 0<=nr<self.m.row and 0<=nc<self.m.column and self.m.map[nr][nc]!=1 and dist[nr][nc]==INF:
                    dist[nr][nc]=dist[r][c]+1
                    q.append((nr,nc))
        return dist

    def _is_corner_deadlock(self,eggs,holes):
        for er,ec in eggs:
            if (er,ec) in holes:
                continue
            up=not self._passable(er-1,ec)
            down=not self._passable(er+1,ec)
            left=not self._passable(er,ec-1)
            right=not self._passable(er,ec+1)
            if (up or down) and (left or right):
                return True
        return False

    def _heuristic(self,eggs,holes):
        total=0
        for e in eggs:
            if e in holes:
                continue
            best=10**9
            for h in holes:
                d=self._hole_dist[h][e[0]][e[1]]
                if d<best:
                    best=d
            if best>=10**9:
                return 10**9
            total+=best
        return total

    def _neighbors(self,mouse,eggs):
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=mouse[0]+dr,mouse[1]+dc
            if nr<0 or nr>=self.m.row or nc<0 or nc>=self.m.column:
                continue
            if self.m.map[nr][nc]==1:
                continue
            if (nr,nc) in eggs:
                br,bc=nr+dr,nc+dc
                if br<0 or br>=self.m.row or bc<0 or bc>=self.m.column:
                    continue
                if self.m.map[br][bc]==1:
                    continue
                if (br,bc) in eggs:
                    continue
                new_eggs=frozenset(e for e in eggs if e!=(nr,nc))|{(br,bc)}
                yield (nr,nc),new_eggs
            else:
                yield (nr,nc),eggs

    def search(self):
        holes=frozenset((int(h[0]),int(h[1])) for h in self.m.hole_list)
        eggs0=frozenset((int(e[0]),int(e[1])) for e in self.m.egg_list)
        start=(int(self.m.row-2),1)

        self._hole_dist={h:self._bfs_distance_map(h) for h in holes}

        h0=self._heuristic(eggs0,holes)
        if h0>=10**9:
            return None

        counter=0
        open_heap=[(h0,0,counter,start,eggs0)]
        came_from={(start,eggs0):None}
        g_score={(start,eggs0):0}

        while open_heap:
            f,g,_,mouse,eggs=heapq.heappop(open_heap)
            self.stats["nodes_expanded"]+=1
            key=(mouse,eggs)

            if eggs==holes:
                path=[key]
                while came_from.get(path[-1]) is not None:
                    path.append(came_from[path[-1]])
                path.reverse()
                return [list(n[0]) for n in path]

            if g_score.get(key,10**9)<g:
                continue

            if self._is_corner_deadlock(eggs,holes):
                continue

            for new_mouse,new_eggs in self._neighbors(mouse,eggs):
                ng=g+1
                nkey=(new_mouse,new_eggs)
                if ng<g_score.get(nkey,10**9):
                    nh=self._heuristic(new_eggs,holes)
                    if nh>=10**9:
                        continue
                    g_score[nkey]=ng
                    came_from[nkey]=key
                    counter+=1
                    heapq.heappush(open_heap,(ng+nh,ng,counter,new_mouse,new_eggs))
        return None

    def main(self,shared_map=None):
        t0=time.perf_counter()
        if shared_map is not None:
            self.load_map(shared_map)
        else:
            self.init_map()
        path=self.search()
        self.complete_path=path if path is not None else []
        self.stats["elapsed"]=time.perf_counter()-t0
        self.stats["steps"]=len(self.complete_path)
        return self.complete_path


if __name__ == "__main__":
    import copy
    m=Map()
    S=Solver(); S.main(shared_map=copy.deepcopy(m))
    T=SokobanSolver(); T.main(shared_map=copy.deepcopy(m))
    print("legacy",S.stats)
    print("sokoban",T.stats)
