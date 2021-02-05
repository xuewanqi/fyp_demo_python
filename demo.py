import pygame
import time
pygame.init()

from data import EASY,MEDIUM,HARD
from graph import Graph
from misc import Misc
from algo import Algo

WINDOW_SIZE = (1200,800)
BACKGROUND = (255,255,255)


#Create the screen
screen = pygame.display.set_mode(WINDOW_SIZE)

#title and icon
pygame.display.set_caption("Demo")

#misc functions
misc = Misc(screen)


def play(data,thief,police,goals):
    #init graph
    graph = Graph(screen,data,thief,police,goals)

    #init algo
    algo = Algo(data[2])

    #main loop
    running = True
    #limit for num of turns
    limit=20
    #gameover var
    gameover = False
    while running:
        screen.fill(BACKGROUND)

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                #player select where to move for thief
                if not gameover:
                    turn=graph.choose(event.pos)
                    if turn:
                        #police move base on algo here
                        graph.police_move(algo.move(graph.thief,graph.police))
                        limit-=1
            
            elif event.type == pygame.QUIT:
                running = False
            
        
        #check who win
        result = graph.checkWin()
        if result ==1: #player win
            misc.message_to_screen('Player wins!',(255,0,0),0,0,'large')
            m= misc.button("Restart",850,320,150,150,(55,255,255),(0,255,0))
            gameover = True
            if m:
                return
        elif result ==2: #police win
            misc.message_to_screen('Player Lose...',(0,255,0),0,0,'large')
            m1= misc.button("Restart",850,320,150,150,(55,255,255),(0,255,0))
            gameover = True
            if m1:
                return
        elif limit==0:
            misc.message_to_screen('Its a draw!',(0,0,255),0,0,'large')
            m2= misc.button("Restart",850,320,150,150,(55,255,255),(0,255,0))
            gameover = True
            if m2:
                return
        else:
            #draw game boundary
            pygame.draw.rect(screen,(0,0,0),(50,50,700,700),1)
            #display graph
            graph.display()
            #display turns left
            misc.message_to_screen('Turns left: '+str(limit),(0,0,255),310,-200,'medium')
        pygame.display.update()

def main():
    #main loop
    running = True
    while running:
        screen.fill(BACKGROUND)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        #select difficulty
        misc.message_to_screen("Select Difficulty",(0,0,0),0,-150,'large')
        e= misc.button("Easy",300,350,150,150,(55,255,255),(0,255,0))
        m= misc.button("Medium",500,350,150,150,(55,255,255),(0,255,0))
        h= misc.button("Hard",700,350,150,150,(55,255,255),(0,255,0))
        if e:
            play(EASY,3,[10],[12])
        if m:
            play(MEDIUM,9,[18,1],[5])
        if h:
            play(HARD,23,[6,15],[30,26])
        pygame.display.update()

main()