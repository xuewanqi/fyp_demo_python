import pygame
police_img = pygame.image.load("./images/police.png")
thief_img = pygame.image.load("./images/robber.png")
exit_img = pygame.image.load("./images/exit.png")

LEFT_BOUNDARY = 100
RIGHT_BOUNDARY = 700 
TOP_BOUNDARY = 100
BOTTOM_BOUNDARY = 700
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLACK = (0,0,0)

class Graph:

    #police given as array of pos
    def __init__(self,screen,data,thief,police,goals):
        self.num_rows = data[0]
        self.num_cols = data[1]
        self.graph = data[2]
        self.num_police = len(police)
        self.police = police
        self.thief = thief
        self.screen = screen
        self.node_width = 600//self.num_cols
        self.node_height = 600//self.num_rows
        self.police_icon = pygame.transform.scale(police_img,(self.node_width//2,self.node_height//2))
        self.thief_icon = pygame.transform.scale(thief_img,(self.node_width//2,self.node_height//2))
        self.exit_icon = pygame.transform.scale(exit_img,(self.node_width//2,self.node_height//2))
        self.police_visited = set(police)
        self.thief_visited = set([thief])
        self.edges={}
        self.goals= set(goals)
        for k,v in self.graph.items():
            for n in v:
                if n!=k:
                    edge = [k,n]
                    edge.sort()
                    edge = tuple(edge)
                    if self.edges.get(edge)==None:
                        self.edges[edge] = BLACK

    def Node(self,x,y,width,height,inactive_color,active_color):
        cur = pygame.mouse.get_pos()
        if x+width > cur[0] > x and y+height > cur[1] > y:
            pygame.draw.rect(self.screen,active_color,(x,y,width,height))
                
        else:
            pygame.draw.rect(self.screen,inactive_color,(x,y,width,height))

    def display(self):
        width = self.node_width//2
        height = self.node_height//2

        #draw edges
        for edge,color in self.edges.items():
            n1,n2 = edge
            n1_y = 100+((n1-1)//self.num_cols)*self.node_height +height
            n1_x = 100+(n1-1-((n1-1)//self.num_cols)*self.num_cols)*self.node_width + width
            n2_y = 100+((n2-1)//self.num_cols)*self.node_height +height
            n2_x = 100+(n2-1-((n2-1)//self.num_cols)*self.num_cols)*self.node_width + width
            pygame.draw.line(self.screen, color, (n1_x,n1_y), (n2_x,n2_y))
            
        #draw nodes
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                pos = i*self.num_cols+j+1
                if pos in self.police_visited:
                    color = GREEN
                elif pos in self.thief_visited:
                    color = RED
                else:
                    color = BLACK
                x = 100 + j*self.node_width + width//2
                y = 100 + i*self.node_height + height//2
                if pos ==self.thief:
                    self.screen.blit(self.thief_icon,(x,y))
                elif pos in self.police:
                    self.screen.blit(self.police_icon,(x,y))
                elif pos in self.goals:
                    self.screen.blit(self.exit_icon,(x,y))
                else:
                    node = self.Node(x,y,width,height,color,BLUE)
        
        
    
    def choose(self,pos):
        x,y = pos
        if x>100 and x<700 and y>100 and y<700:
            x-=100
            y-=100
            w = self.node_width//2
            h = self.node_height//2
            i = y//self.node_height
            j = x//self.node_width
            p = i*self.num_cols+j+1
            if p in self.graph[self.thief]:
                self.thief_visited.add(p)
                edge = [p,self.thief]
                edge.sort()
                edge= tuple(edge)
                self.edges[edge] = RED
                self.thief = p
                return True
        return False

    def police_move(self,pos):
        for i in range(len(pos)):
            edge = [pos[i],self.police[i]]
            self.police_visited.add(pos[i])
            edge.sort()
            edge = tuple(edge)
            self.edges[edge] = GREEN
            self.police = pos

                
    #check who win
    def checkWin(self):
        if self.thief in self.goals:
            return 1
        elif self.thief in self.police:
            return 2
        return 0
                




    

    

