#!/usr/bin/env python
from numpy import pi
from time import sleep
from rltoolbox.environment.models import *

def animate(model, action):
    model.step(action)
    model.render()
    print(model.observation)
    sleep(model.timestep)

def demo_ball_beam():
    bb = BallBeam()
    bb.reset()
    animate(bb, -pi/8)
    animate(bb, -pi/4)
    animate(bb, pi/4)
    for i in range(10):
        animate(bb, pi/4)
    for i in range(20):
        animate(bb, -pi/4)
    for i in range(40):
        animate(bb, pi/8)
    bb.close()

def demo_mountain_car():
    cc = MountainCar()
    cc.reset()
    for i in range(20):
        animate(cc, -1.0)
    for i in range(30):
        animate(cc, 1.0)
    for i in range(30):
        animate(cc, -1.0)
    for i in range(100):
        animate(cc, 1.0)
    cc.close()

def demo_cart_pole():
    cp = CartPole()
    cp.reset()
    for i in range(100):
        if i < 5:
            animate(cp, 10.0)
        elif 5 <= i < 20:
            animate(cp, -10.0)
        elif 20 <= i < 100:
            animate(cp, 10.0)
    cp.close()


if __name__ == "__main__":
    demo_ball_beam()
    demo_mountain_car()
    demo_cart_pole()
