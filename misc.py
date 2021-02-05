import pygame

smallfont = pygame.font.SysFont('calibri',25)
mediumfont = pygame.font.SysFont('calibri',50)
largefont = pygame.font.SysFont('calibri',80)
WINDOW_SIZE = (1200,800)

class Misc:
    def __init__(self,screen):
        self.screen = screen

    def text_objects(self, text, color ,size ):
        if size == 'small':
            textsurface = smallfont.render(text,True,color)
        elif size == 'medium':
            textsurface = mediumfont.render(text,True,color)
        elif size == 'large':
            textsurface = largefont.render(text,True,color)
        return textsurface, textsurface.get_rect()

    def message_to_screen(self,msg,color,x_displace,y_displace,size):
        textsurf, textrect = self.text_objects(msg,color,size)
        textrect.center = (WINDOW_SIZE[0]/2)+x_displace,(WINDOW_SIZE[1]/2)+ y_displace
        self.screen.blit(textsurf,textrect)
    
    def text_to_button(self,msg,color,buttonx,buttony,buttonwidth,buttonheight,size):
        textsurf, textrect = self.text_objects(msg,color,size)
        textrect.center = ((buttonx +(buttonwidth/2)), buttony+(buttonheight/2))
        self.screen.blit(textsurf,textrect)

    def button(self,text,x,y,width,height,inactive_color,active_color):
        a = 0
        cur = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if x+width > cur[0] > x and y+height > cur[1] > y:
            pygame.draw.rect(self.screen,active_color,(x,y,width,height))
            if click[0] == 1:
                a= 1
                
        else:
            pygame.draw.rect(self.screen,inactive_color,(x,y,width,height))
        self.text_to_button(text,(0,0,0),x,y,width,height,'small')
        return a