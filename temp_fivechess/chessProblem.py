# 五子棋问题
from graphics import *
import time

class Plaer:
    def __init__(self) -> None:
        pass 

    def action(problem):
        pass 

class chess:
    def __init__(self,playerA,playerB) -> None:
        self.DONE = False
        self.WHOFIRST = 1
        self.EPOCH = 1
        self.L1MAX = -10000
        self.L2MIN = 10000
        self.TRACK = []
        
        self.RESTART_FLAG = False
        self.QUIT_FLAG = False
        self.START_FLAG = False

        self.RESTART = Text(Point(500,60),"restart")
        self.QUIT = Text(Point(500,20),"quit")

        self.num = [[0 for a in range(16)] for a in range(16)]
        self.palyerA = Text(Point(500,100),"")
        self.palyerB = Text(Point(500,140),"")
        # chess board
        self.notice = Text(Point(500,290),"") #提示轮到谁落子
        self.notice.setFill('red')
        # chess board
        self.chessboard = GraphWin("Chess borad", 550, 451)
        self.chessboard.setBackground("burlywood")
        self.coord_A = Text(Point(500,330),"") #AI最后落子点
        self.coord_B = Text(Point(500,370),"") #玩家最后落子点
        # draw the chess borad
        for i in range(0,451,30):
            line = Line(Point(0,i),Point(450,i))
            line.draw(self.chessboard)
        for i in range(0,451,30):
            line = Line(Point(i,0),Point(i,450))
            line.draw(self.chessboard)
        # draw the coordinate
        Rectangle(Point(460,5),Point(540,35)).draw(self.chessboard)
        Rectangle(Point(460,45),Point(540,75)).draw(self.chessboard)
        Rectangle(Point(460,85),Point(540,115)).draw(self.chessboard)
        Rectangle(Point(460,125),Point(540,155)).draw(self.chessboard)
        Rectangle(Point(452,275),Point(548,305)).draw(self.chessboard)
        Rectangle(Point(452,307),Point(548,395)).draw(self.chessboard)
        self.palyerA.draw(self.chessboard)
        self.palyerB.draw(self.chessboard)
        self.notice.draw(self.chessboard)
        self.coord_A.draw(self.chessboard)
        self.coord_B.draw(self.chessboard)
        self.QUIT.draw(self.chessboard)
        self.RESTART.draw(self.chessboard)
    
    def reset(self):
        # chess board
        self.DONE = False
        self.WHOFIRST = 1
        self.EPOCH = 1
        self.RESTART_FLAG = False
        self.QUIT_FLAG = False
        for i in range(16):
            for j in range(16):
                if self.num[i][j] != 0:
                    self.num[i][j] = 0
        for i in range(len(self.TRACK)):
            self.TRACK[-1].undraw()
            self.TRACK.pop()
        self.palyerA.setText("Palyer A")
        self.palyerB.setText("Palyer B")
        self.notice.setText("")
        self.coord_A.setText("")
        self.coord_B.setText("")
        return True

    def step(self,coord):
        '''
        According to the coord, there is some state need to transit
        state1: self.RESTART_FLAG
        state2: self.QUIT_FLAG
        state3: self.WHOFIRST
        state4: self.DONE
        state5: transit
        '''
        # coord: (x,y)
        x = coord.getX()
        y = coord.getY()
        if((abs(500-x)<40) and (abs(60-y)<15)): #restart
            self.reset()
            self.RESTART_FLAG=True
            self.notice.setText("Restart_again")
            time.sleep(1)
            print('INFO: the player choose to restart')
            return
        elif((abs(500-x)<40) and (abs(20-y)<15)): #quit
            self.QUIT_FLAG=True
            print('INFO: the player choose to quit')
            self.reset()
            self.DONE = True
            self.notice.setText("Quit")
            time.sleep(1)
            return

        if((abs(500-x)<40) and (abs(100-y)<15)): #AI 先手
            self.EPOCH=1
            self.WHOFIRST=1
            self.palyerA.setText("playerA black")
            self.palyerB.setText("playerB white")
            self.START_FLAG = True
            return
        
        if((abs(500-x)<40) and (abs(140-y)<15)): #玩家先手
            self.WHOFIRST=2
            self.palyerA.setText("playerA white")
            self.palyerB.setText("playerB black")
            self.START_FLAG = True
            return
        print('INFO : the player choose to step')
        

        positionX = round(x/30)
        positionY = round(y/30)
        newChess = Circle(Point(positionX*30,positionY*30),13)
        if self.num[positionX][positionY] != 0:
            self.notice.setText("This position is occupied")
            time.sleep(1)
            return self.DONE
        
        if positionX<16 and positionX>-1 and positionY<16 and positionY>-1:
            if (self.EPOCH+self.WHOFIRST)%2 == 1:
                # Player A
                self.EPOCH+=1
                newChess.setFill('black')
            else:
                # Plaer B
                self.EPOCH+=1
                newChess.setFill('white')
            newChess.draw(self.chessboard)
            self.TRACK.append(newChess)   
            self.num[positionX][positionY] = self.EPOCH
        else:
            # 超出边界
            self.notice.setText("Out of range")
        
        print('INFO: the player step at (%d,%d)'%(positionX,positionY))
        
        return self.DONE
    def isGoalState():
        '''
        判断是否是目标状态
        '''
        pass
    def showWhoFirst(self):
        '''
        according to the self.WHOFIRST, show who first
        '''
        if self.WHOFIRST==1:
            self.notice.setText("Player A first")
        else:
            self.notice.setText("Palyer B first")

game = chess(Plaer(),Plaer())
game.reset() 
while not game.START_FLAG and not game.QUIT_FLAG:
    p = game.chessboard.getMouse()
    game.step(p)
print('Start game')
while not game.DONE:
    game.RESTART_FLAG = False
    p = game.chessboard.getMouse()
    game.step(p)
    game.showWhoFirst()