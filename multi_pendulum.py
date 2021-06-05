import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from math import pi

from manim import *
import os
import pyclbr

color = [BLUE, GREEN, ORANGE, PURPLE, YELLOW]

def trivial_ode(f, t0, x0, ti, h = 1):
    T = [t0]
    X = [x0]
    while T[-1] < ti:
        x, t = X[-1], T[-1]
        T.append(t + h)
        X.append(x + h * f(x, t))
    return np.array(T), np.array(X)

def rk4(f, t0, x0, ti, h = 1):
    T = [t0]
    X = [x0]
    while T[-1] < ti:
        x, t = X[-1], T[-1]
        k1 = h * f(x, t)
        k2 = h * f(x + 0.5*k1, t + 0.5*h)
        k3 = h * f(x + 0.5*k2, t + 0.5*h)
        k4 = h * f(x + k3, t + h)
        T.append(t + h)
        X.append(x + (k1 + 2*k2 + 2*k3 + k4)/6)
    return np.array(T), np.array(X)

def leapfrog(f, t0, x0, ti, h = 1):
    T = [t0]
    X = [x0]
    x_half = x0 + 0.5 * h * f(x0, t0)
    while T[-1] < ti:
        x, t = X[-1], T[-1]
        T.append(t + h)
        X.append(x + h * f(x_half, t + 0.5*h))
        x_half += h * f(X[-1], T[-1])
    return np.array(T), np.array(X)

# xi = [phi_0, phi_1, ..., phi_n, phi_dot_0, phi_dot_1, ..., phi_dot_n]
g = 9.8

class temp_updater_dot:
    def __init__(self, X1, step = 0.08) -> None:
        self.time = 0
        self.n = 0
        self.X1 = X1
        self.step = step
    def update_func(self, mobject, dt):
        self.n += 1
        self.time += dt
        s = int(self.time / self.step)
        mobject.move_to(self.X1[s])

class temp_updater_line:
    def __init__(self, X1, X2, step = 0.08) -> None:
        self.time = 0
        self.n = 0
        self.X1 = X1
        self.X2 = X2
        self.step = step
    def update_func(self, mobject, dt):
        self.n += 1
        self.time += dt
        s = int(self.time / self.step)
        mobject.put_start_and_end_on(self.X1[s], self.X2[s])

class multi_pendulum(Scene):
    def init(self, N, R, M, Phi0, ode = leapfrog, step = 0.01) -> None:
        self.N = N
        self.R = R
        self.M = M
        self.Phi0 = Phi0
        self.xi0 = np.zeros(2*N)
        self.xi0[:N] = np.array(Phi0)
        self.ode = ode
        self.step = step
    
    def diff_xi(self, xi, t):
        N = self.N
        diff_xi = np.zeros(2*N)
        diff_xi[:N] = xi[N:]
        phi = xi[:N]
        phi_dot = xi[N:]
        m = self.M
        R = self.R

        a = np.zeros([N, N])
        b = np.zeros(N)
        for k in range(N):
            for i in range(k, N):
                b[k] += -g * m[i] * np.sin(phi[k])
                for j in range(i + 1):
                    a[k, j] += m[i] * R[j] * np.cos(phi[j] - phi[k])
                    b[k] += m[i] * R[j] * (phi_dot[j]**2) * np.sin(phi[j] - phi[k])
        diff_xi[N:] = np.linalg.solve(a, b)
        #print(a, b)

        return diff_xi
    
    def Ek(self, xi):
        N = self.N
        phi = xi[:N]
        phi_dot = xi[N:]
        m = self.M
        R = self.R

        K = 0
        vx = vy = 0
        for i in range(N):
            vx += R[i] * phi_dot[i] * np.cos(phi[i])
            vy += R[i] * phi_dot[i] * np.sin(phi[i])
            K += 0.5 * m[i] * (vx**2 + vy**2)

        return K
    
    def V(self, xi):
        N = self.N
        phi = xi[:N]
        m = self.M
        R = self.R

        V = 0
        m_acc = np.sum(m)
        for i in range(N):
            V += -g * m_acc * R[i] * np.cos(phi[i])
            m_acc -= m[i]

        return V
    
    def Lagrangian(self, xi):
        return self.Ek(xi) - self.V(xi)
    
    def Hamiltonian(self, xi):
        return self.Ek(xi) + self.V(xi)

    def simulate(self, t0, t1, h = 0.1):
        T, Xi = self.ode(self.diff_xi, t0, self.xi0, t1, h)
        return T, Xi
    
    def plot_trajectory(self):
        L = np.sum(self.R)
        T, Xi = self.simulate(0, 30, self.step)
        plt.figure(figsize=(6,6))
        X, Y = 0, 0
        for i in range(self.N):
            X += self.R[i]*np.sin(Xi[:,i])
            Y -= self.R[i]*np.cos(Xi[:,i])
            plt.plot(X, Y, label=r"$ball_{}$".format(i))
        
        plt.scatter(0, 0)
        plt.xlim(-L*1.1, L*1.1)
        plt.ylim(-L*1.6, L*0.6)
        plt.xticks((-L, 0, L))
        plt.yticks((-L, 0))
        #plt.show()
    
    def return_energy(self):
        T, Xi = self.simulate(0, 30, self.step)
        E = []
        for xi in Xi:
            E.append(self.Hamiltonian(xi))
        return T, E
    
    def construct(self):
        self.init(5, np.array([0.8, 0.5, 0.3, 0.2, 0.1])*2, [1, 1, 1, 1, 1], [pi/2, pi/6, -pi/3, -pi/2, 0], ode=rk4, step=0.01)
        T, Xi = self.simulate(0, 32, self.step)

        offset = UP*2
        X = [np.array([offset] * len(Xi))]
        for i in range(self.N):
            X.append(np.array([np.sin(Xi[:,i])*self.R[i], -np.cos(Xi[:,i])*self.R[i], np.zeros(len(Xi))]).transpose() + X[i])

        Pivot = Dot(color = RED).shift(offset)

        L = []
        for i in range(self.N):
            L.append(Line())
            L[-1].add_updater(temp_updater_line(X[i], X[i + 1], step = self.step).update_func)
            self.add(L[-1])

        self.add(Pivot)
        
        B = []
        for i in range(self.N):
            B.append(Dot(color = color[i]))
            B[-1].add_updater(temp_updater_dot(X[i + 1], step = self.step).update_func)
            self.add(B[-1])

        self.wait(30)

# P = multi_pendulum(5, [8, 5, 3, 2, 1], [1, 1, 1, 1, 1], [pi/2, pi/6, -pi/3, -pi/2, 0], ode=rk4, step=0.01)
