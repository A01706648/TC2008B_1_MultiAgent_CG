# Importamos las clases que se requieren para manejar los agentes (Agent) y su entorno (Model).
# Cada modelo puede contener múltiples agentes.
from mesa import Agent, Model 

# Debido a que necesitamos que existe un solo agente por celda, elegimos ''SingleGrid''.
from mesa.space import SingleGrid

# Con ''RandomActivation'', hacemos que todos los agentes se activen ''al mismo tiempo''.
from mesa.time import RandomActivation
from mesa.time import BaseScheduler

# Haremos uso de ''DataCollector'' para obtener información de cada paso de la simulación.
from mesa.datacollection import DataCollector

# matplotlib lo usaremos crear una animación de cada uno de los pasos del modelo.
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"
matplotlib.rcParams['animation.embed_limit'] = 2**128

# Importamos los siguientes paquetes para el mejor manejo de valores numéricos.
import numpy as np
import pandas as pd

# Definimos otros paquetes que vamos a usar para medir el tiempo de ejecución de nuestro algoritmo.
import time
import datetime

from random import randrange


#Constant
DIR_UP = 0
DIR_DOWN = 1
DIR_LEFT = 2
DIR_RIGHT = 3

STATE_WAIT = 0
STATE_INTER = 1
STATE_LEAVE = 2

#Setting
TRIGGER_PERIOD_MS_TRAFFIC = 1
TRIGGER_PERIOD_MS_CAR = 1
CAR_INCREASE_PERIOD = 1
INTERVAL_TIME_S = 0.1

LANE_WIDTH = 2.0
MAX_CAR_WAIT = 20
MIN_CAR_LENGTH = 3.0
MAX_CAR_LENGTH = 3.0
SAFE_DISTANCE = 0.5
MAX_CAR_WAITLINE = 10
MAX_CAR_SPEED = (60 / 3.6) * INTERVAL_TIME_S #km/h

MAX_STEP = 100

WAIT_LINE_NUM = 4

MAP_WIDTH = (LANE_WIDTH * 2) + (MAX_CAR_WAITLINE * (MAX_CAR_LENGTH + SAFE_DISTANCE))
MAP_HEIGHT = MAP_WIDTH


class TrafficAgent(Agent):
  def __init__(self, unique_id, model, fLaneWidth, uCarIncreasePeriodMs, uMaxCarInWaitLine):
    super().__init__(unique_id, model)
    self.fLaneWidth = fLaneWidth
    self.fInterSize = 4 * fLaneWidth
    self.uCarIncreasePeriodMs = uCarIncreasePeriodMs
    self.uMaxCarInWaitLine = uMaxCarInWaitLine
    self.CarLists = list()
    self.CarlineList = list()
    self.uStepCount = 0

    #create car lists
    for count in range(WAIT_LINE_NUM):
      self.CarLists.append(list())

    #add car lines
    for count in range(WAIT_LINE_NUM):
      Carline = None
      if((count % 2) == 0):
        Carline = CarlineModel(self, uCarIncreasePeriodMs, uMaxCarInWaitLine, count, (count + 1))
      else:
        Carline = CarlineModel(self, uCarIncreasePeriodMs, uMaxCarInWaitLine, count, (count - 1))

      self.CarlineList.append(Carline)
    

  def step(self):
    #print("traffic agent")
    self.decideGreenLight()

  def getInitPosition(self, CarLineIndex):
    CarLine = self.CarlineList[CarLineIndex]
    result = MAP_WIDTH
    if(len(CarLine.schedule.agents) > 0):
      LastCar = CarLine.schedule.agents[-1]
      position = LastCar.getAbsPosition()
      if(position >= MAP_WIDTH - SAFE_DISTANCE):
        result += position + LastCar.fCarSize  + SAFE_DISTANCE
      else:
        result = MAP_WIDTH
    return result
    
  def decideGreenLight(self):
    self.uStepCount += 1
    for count in range(4):
      if(self.uStepCount % 10 % 4 == count):
        self.CarlineList[count].setGreen(True)
      else:
        self.CarlineList[count].setGreen(False)


class TrafficModel(Model):
  def __init__(self, fLaneWidth, uCarIncreasePeriodMs, uMaxCarInWaitLine):
    self.schedule = BaseScheduler(self)
    self.traffic = TrafficAgent(0, self, fLaneWidth, uCarIncreasePeriodMs, uMaxCarInWaitLine)
    self.schedule.add(self.traffic)
    self.datacollector = DataCollector(model_reporters = {"Particles" : get_cars})
    self.count_step = 0

  def step(self):
    #print("traffic model")

    #schedule traffic
    if(self.count_step % TRIGGER_PERIOD_MS_TRAFFIC == 0):
      self.schedule.step()
    #schedule carline  
    if(self.count_step % TRIGGER_PERIOD_MS_CAR == 0):
      for Carline in self.traffic.CarlineList:
        #add new car
        if(len(Carline.schedule.agents) < Carline.uMaxCarInWaitLine):
          if(self.count_step % Carline.fCarIncreasePeriodS == 0):
            #first step, add car anyway
            if((randrange(2) == 1) or (self.count_step == 0)):
              Carline.addCar(Carline.FromDirection, Carline.ToDirection)
        else:
          pass#reach to maxium car line
        Carline.step()

    #count up step
    self.count_step += 1

    #get the position of cars
    self.datacollector.collect(self)


class CarAgent(Agent):
  def __init__(self, unique_id, model, fCarSize, fSafeDistance, FromDirection, ToDirection):
    super().__init__(unique_id, model)
    #local member
    self.id = unique_id
    self.FromDirection = FromDirection
    self.ToDirection = ToDirection
    self.pTraffic = model.pTraffic
    self.fCarSize = fCarSize
    self.fSafeDistance = fSafeDistance
    self.state = STATE_WAIT

    #represent the position of the agent in 2d vector
    #add initial position
    initialPosition = self.pTraffic.getInitPosition(FromDirection)# + fSafeDistance
    x_p = 0
    y_p = 0
    x_s = 0
    y_s = 0
    if(FromDirection == DIR_UP):
      x_p = -self.pTraffic.fLaneWidth
      y_p = initialPosition
    elif(FromDirection == DIR_DOWN):
      x_p = self.pTraffic.fLaneWidth
      y_p = -initialPosition
    elif(FromDirection == DIR_LEFT):
      x_p = -initialPosition
      y_p = -self.pTraffic.fLaneWidth
    elif(FromDirection == DIR_RIGHT):
      x_p = initialPosition
      y_p = self.pTraffic.fLaneWidth
    else:
      pass

    self.position = np.array((x_p, y_p), dtype=np.float64)
    self.speed = np.array((x_s, y_s), dtype=np.float64)
    #set to go
    self.setSpeed(True)

    #Update State
    self.updateState()


  def step(self):
    #print("Car agent")

    #in the end of the step
    self.updateState()
    if(self.state == STATE_WAIT and self.getAbsPosition() < MIN_CAR_LENGTH):#in the first line
      if(self.model.bIsGreen):
        self.setSpeed(True)
      else:
        self.setSpeed(False)
    else:#not in the first line
      frontDistant = self.getDistanceFrontCar()
      #if(frontDistant <= self.fSafeDistance):
      if(frontDistant <= 3 * MAX_CAR_SPEED * INTERVAL_TIME_S):
        self.setSpeed(False)
        #print("stop")
      else:
        #print(frontDistant)          
        self.setSpeed(True)
    self.position = self.position + self.speed
    self.updateState()

    if(False):
      if(self.FromDirection == DIR_UP):
        if(self.unique_id == 0):
          print("0: " + str(self.getAbsPosition()))
        if(self.unique_id == 1):
          print("1: " + str(self.getAbsPosition()))  
          print("D: " + str(self.getDistanceFrontCar()))

    if(self.isOutMap()):
      self.model.schedule.remove(self)
    #if(self.id == 0 and self.FromDirection == DIR_UP):
    #  print(self.position)

  def updateState(self):
    pos = self.getAbsPosition()
    if(pos > self.pTraffic.fLaneWidth):
      self.state = STATE_WAIT
    elif(pos < -self.pTraffic.fLaneWidth):
      self.state = STATE_LEAVE
    else:
      self.state = STATE_INTER

  def setSpeed(self, isGo):
    speed = 0
    if(isGo):
      speed = MAX_CAR_SPEED

    x_s = 0
    y_s = 0

    if(self.FromDirection == DIR_UP):
      x_s = 0
      y_s = -speed
    elif(self.FromDirection == DIR_DOWN):
      x_s = 0
      y_s = speed
    elif(self.FromDirection == DIR_LEFT):
      x_s = speed
      y_s = 0
    elif(self.FromDirection == DIR_RIGHT):
      x_s = -speed
      y_s = 0
    else:
      pass

    self.speed = np.array((x_s, y_s), dtype=np.float64)
    
  def isOutMap(self):
    result = False
    #if(abs(self.position[0]) > MAP_WIDTH or abs(self.position[1]) > MAP_HEIGHT):
    #  result = True
    if(False):
      if(self.FromDirection == DIR_UP):
        if(self.position[1] < -MAP_HEIGHT):
          result = True
      elif(self.FromDirection == DIR_DOWN):
        if(self.position[1] > MAP_HEIGHT):
          result = True
      elif(self.FromDirection == DIR_LEFT):
        if(self.position[0] > MAP_WIDTH):
          result = True
      elif(self.FromDirection == DIR_RIGHT):
        if(self.position[0] < -MAP_WIDTH):
          result = True
      else:
        pass
    else:
      if(self.getAbsPosition() < -MAP_HEIGHT):
        result = True

    return result

  def getAbsPosition(self):
    result = 0
    if(self.FromDirection == DIR_UP):
      result = self.position[1]#y
    elif(self.FromDirection == DIR_DOWN):
      result = -self.position[1]#-y
    elif(self.FromDirection == DIR_LEFT):
      result = -self.position[0]#-x
    elif(self.FromDirection == DIR_RIGHT):
      result = self.position[0]#x
    else:
      pass

    return result

  def getDistanceFrontCar(self):
    line = self.model
    traffic = self.pTraffic
    car_list = self.model.schedule.agents
    distance = 0xffff
    myPos = self.getAbsPosition()

    for car in car_list:
      otherPos = car.getAbsPosition()
      if(otherPos <= myPos and car.unique_id != self.unique_id):
        if((myPos - otherPos) <= distance):
          distance = myPos - otherPos - car.fCarSize
          #print(distance)
    
    return distance    

class CarlineModel(Model):
  def __init__(self, pTraffic, fCarIncreasePeriodS, uMaxCarInWaitLine, FromDirection, ToDirection):
    self.schedule = BaseScheduler(self)
    self.fCarIncreasePeriodS = fCarIncreasePeriodS
    self.uMaxCarInWaitLine = uMaxCarInWaitLine
    self.uCarCount = 0
    self.FromDirection = FromDirection
    self.ToDirection = ToDirection
    self.pTraffic = pTraffic
    self.bIsGreen = False

  def getCarSize(self):
    return MIN_CAR_LENGTH

  def addCar(self, FromDirection, ToDirection):
    car = CarAgent(self.uCarCount, self, self.getCarSize(), SAFE_DISTANCE, FromDirection, ToDirection)
    self.schedule.add(car)
    self.uCarCount += 1
    self.pTraffic.CarLists[FromDirection].append(car)

  def step(self):
    #print("Carline Model")
    self.schedule.step()

  def setGreen(self, bIsGreen):
    self.bIsGreen = bIsGreen        



def get_cars(model):
  result = []
  for agent in model.schedule.agents:
    #for carline in agent.CarlineList:
    #  for car in carline.schedule.agents:
    for carlist in agent.CarLists:
      for car in carlist:
        #result.append(np.append(car.position, [car.state], axis = 0))
        result.append(car.position)
  result = np.asarray(result)
  return result        


if __name__ == "__main__":
  traffic_mode = TrafficModel(LANE_WIDTH, CAR_INCREASE_PERIOD, MAX_CAR_WAITLINE)
  for i in range(MAX_STEP):
      traffic_mode.step()
      #time.sleep(0.001)  

  all_positions = traffic_mode.datacollector.get_model_vars_dataframe()    

  fig, ax = plt.subplots(figsize=(7,7))
  scatter = ax.scatter(all_positions.iloc[0][0][:,0], all_positions.iloc[0][0][:,1], 
                    s=10, cmap="jet", edgecolor="k")

  ax.axis([-MAP_WIDTH, MAP_WIDTH, -MAP_HEIGHT, MAP_HEIGHT])

  midline_v = ax.vlines(x=0, ymin=-MAP_HEIGHT, ymax=MAP_HEIGHT, color='r')
  midline_h = ax.hlines(y=0, xmin=-MAP_WIDTH, xmax=MAP_WIDTH, color='r')

  def update(frame_number):
      global midline_v
      #global scatter
      scatter.set_offsets(all_positions.iloc[frame_number][0])
      #scatter.set_color('r')
      #if(frame_number % 2):
      #  scatter.remove()
      #else:
      #  scatter = ax.scatter(all_positions.iloc[frame_number][0][:,0], all_positions.iloc[frame_number][0][:,1], 
      #              s=10, cmap="jet", edgecolor="k")
      #for pos in all_positions.iloc[frame_number][0]:
      #  color = 'k'
      #  if(pos[2] == STATE_WAIT):
      #    color = 'r'
      #  elif(pos[2] == STATE_INTER): 
      #    color = 'g'
      #  else:
      #    color = 'b'  
      #  ax.scatter(pos[0], )


      return scatter#,midline_v

  anim = animation.FuncAnimation(fig, update, frames=MAX_STEP)

  plt.show()