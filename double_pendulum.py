import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim
from math import pi

def trivial_ode(f, t0, x0, ti, h = 1):
    T = [t0]
    X = [x0]
    while T[-1] < ti:
        x, t = X[-1], T[-1]
        T.append(t + h)
        X.append(x + h * f(x, t))
    return np.array(T), np.array(X)

def rk2(f, t0, x0, ti, h = 1):
    T = [t0]
    X = [x0]
    while T[-1] < ti:
        x, t = X[-1], T[-1]
        k1 = h * f(x, t)
        k2 = h * f(x + 0.5*k1, t + 0.5*h)
        T.append(t + h)
        X.append(x + k2)
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

# xi = [phi_0, phi_1, phi_dot_0, phi_dot_1]
g = 9.8

class double_pendulum:
    def __init__(self, R, M, Phi0, ode = leapfrog, step = 0.01) -> None:
        self.R = R
        self.M = M
        self.Phi0 = Phi0
        self.xi0 = np.array([Phi0[0], Phi0[1], 0, 0])
        self.ode = ode
        self.step = step
    
    def diff_xi(self, xi, t):
        diff_xi = np.array([xi[2], xi[3], 0, 0])
        phi = np.array([xi[0], xi[1]])
        phi_dot = np.array([xi[2], xi[3]])
        m = self.M
        R = self.R

        a = (m[0]+m[1])*(R[0]**2)
        b = c = m[1]*R[0]*R[1]*np.cos(phi[0]-phi[1])
        d = m[1]*(R[1]**2)
        e = -m[1]*R[0]*R[1]*(phi_dot[1]**2)*np.sin(phi[0]-phi[1])-g*(m[0]+m[1])*R[0]*np.sin(phi[0])
        f = m[1]*R[0]*R[1]*(phi_dot[0]**2)*np.sin(phi[0]-phi[1])-g*m[1]*R[1]*np.sin(phi[1])

        D = a*d - b*c
        diff_xi[2] = (d*e - b*f) / D
        diff_xi[3] = (a*f - c*e) / D
        return diff_xi
    
    def Ek(self, xi):
        phi = np.array([xi[0], xi[1]])
        phi_dot = np.array([xi[2], xi[3]])
        m = self.M
        R = self.R
        return m[0]/2*(R[0]*phi_dot[0])**2 + m[1]/2*((R[0]*phi_dot[0])**2 + (R[1]*phi_dot[1])**2 \
            + 2*R[0]*R[1]*phi_dot[0]*phi_dot[1]*np.cos(phi[0] - phi[1]))
    
    def V(self, xi):
        phi = np.array([xi[0], xi[1]])
        m = self.M
        R = self.R
        return -g*(m[0]*R[0]*np.cos(phi[0]) + m[1]*R[0]*np.cos(phi[0]) + m[1]*R[1]*np.cos(phi[1]))
    
    def Lagrangian(self, xi):
        return self.Ek(xi) - self.V(xi)
    
    def Hamiltonian(self, xi):
        return self.Ek(xi) + self.V(xi)

    def simulate(self, t0, t1, h = 0.1):
        T, Xi = self.ode(self.diff_xi, t0, self.xi0, t1, h)
        return T, Xi
    
    def energy_verification(self, T, Xi, h):
        X0, Y0 = self.R[0]*np.sin(Xi[:,0]) , -self.R[0]*np.cos(Xi[:,0])
        X1, Y1 = X0 + self.R[1]*np.sin(Xi[:,1]), Y0 - self.R[1]*np.cos(Xi[:,1])
        diff_X0, diff_Y0, diff_X1, diff_Y1 = np.diff(X0)/h, np.diff(Y0)/h, np.diff(X1)/h, np.diff(Y1)/h
        E = g*self.M[0]*Y0[1:] + g*self.M[1]*Y1[1:] + 0.5*self.M[0]*(diff_X0**2+diff_Y0**2) + 0.5*self.M[1]*(diff_X1**2+diff_Y1**2)
        return T[:1], E

    def plot_trajectory(self):
        T, Xi = self.simulate(0, 10, self.step)
        X0, Y0 = self.R[0]*np.sin(Xi[:,0]) , -self.R[0]*np.cos(Xi[:,0])
        X1, Y1 = X0 + self.R[1]*np.sin(Xi[:,1]), Y0 - self.R[1]*np.cos(Xi[:,1])
        
        plt.figure(figsize=(6,6))
        plt.scatter(0, 0)
        plt.plot(X0, Y0)
        plt.plot(X1, Y1)
        plt.xlim(-4, 4)
        plt.ylim(-6, 2)
        plt.show()

    def plot_trajectory_illustrate(self):
        T, Xi = self.simulate(0, 10, self.step)
        X0, Y0 = self.R[0]*np.sin(Xi[:,0]) , -self.R[0]*np.cos(Xi[:,0])
        X1, Y1 = X0 + self.R[1]*np.sin(Xi[:,1]), Y0 - self.R[1]*np.cos(Xi[:,1])
        
        plt.figure(figsize=(6,6))
        plt.plot(X1, Y1, color='lightblue')
        plt.plot([0, X0[-1]], [0, Y0[-1]], color='white')
        plt.plot([X0[-1], X1[-1]], [Y0[-1], Y1[-1]], color='white')
        plt.plot(0, 0, color='red', marker='o')
        plt.plot(X0[-1], Y0[-1], color='purple', marker='o')
        plt.plot(X1[-1], Y1[-1], color='blue', marker='o')
        plt.xlim(-4, 4)
        plt.ylim(-6, 2)
        plt.axis("off")
        plt.savefig("illustrate.png", dpi=750, transparent=True)
        plt.show()
    
    def return_energy(self):
        T, Xi = self.simulate(0, 30, self.step)
        #X0, Y0 = self.R[0]*np.sin(Xi[:,0]) , -self.R[0]*np.cos(Xi[:,0])
        #Tp, E = self.energy_verification(T, Xi, 0.01)
        #plt.plot(Tp,E)
        E = []
        for xi in Xi:
            E.append(self.Hamiltonian(xi))
        return T, E
        #plt.plot(T, E)
        #plt.show()
    
    def anim_config(self, fps = 100):
        T, Xi = self.simulate(0, 30, self.step)
        self.anim_T = T
        self.anim_Phi = Xi[:,0:2]
        self.anim_fps = fps
        self.anim_duration = T[-1] - T[0]
        self.anim_fig, self.anim_ax = plt.subplots(figsize=(6, 6))
        self.pivot, = plt.plot(0, 0, 'o', animated=True)
        self.balls = []
        self.bars = []
        self.L = np.sum(self.R)
        self.time = plt.text(-self.L*1.0, self.L*0.5, r"")
        previous_point = np.array([0., 0.])
        for i in range(len(self.R)):
            point = np.array([self.R[i]*np.sin(self.Phi0[i]) , -self.R[i]*np.cos(self.Phi0[i])]) + previous_point
            self.balls.append(plt.plot(point[0], point[1], 'o', animated=True)[0])
            self.bars.append(plt.plot([previous_point[0], point[0]], [previous_point[1], point[1]], '-', animated=True)[0])
            previous_point = point
        
    def anim_init(self):
        L = self.L
        self.anim_ax.set_xlim(-L*1.1, L*1.1)
        self.anim_ax.set_ylim(-L*1.6, L*0.6)
        self.anim_ax.set_xticks((-L, 0, L))
        self.anim_ax.set_yticks((-L, 0))
        return self.pivot, self.time, self.balls[0], self.balls[1], self.bars[0], self.bars[1]
    
    def anim_update(self, frame):
        previous_point = np.array([0., 0.])
        self.time = plt.text(-self.L*1.0, self.L*0.5, r"$t={:.2f}$s".format(frame/self.anim_fps))
        for i in range(len(self.R)):
            point = np.array([self.R[i]*np.sin(self.anim_Phi[frame][i]) , -self.R[i]*np.cos(self.anim_Phi[frame][i])]) + previous_point
            self.balls[i].set_data(point[0], point[1])
            self.bars[i].set_data([previous_point[0], point[0]], [previous_point[1], point[1]])
            previous_point = point
        return self.pivot, self.time, self.balls[0], self.balls[1], self.bars[0], self.bars[1]
    
    def anim_plot(self):
        a = anim.FuncAnimation(self.anim_fig, self.anim_update, frames=np.arange(0, len(self.anim_T), 1),interval=self.step*1000,\
                    init_func=self.anim_init,blit=True)
        #a.save('p3.mp4',writer='ffmpeg',fps=100)
        plt.show()
'''
P = double_pendulum([2, 1], [2, 1], [pi/2, pi/6], ode=trivial_ode, step=0.01)
plt.plot(*P.return_energy(), label=r"Euler,$h=0.01$s")
P = double_pendulum([2, 1], [2, 1], [pi/2, pi/6], ode=rk2, step=0.08)
plt.plot(*P.return_energy(), label=r"rk2,$h=0.08$s")
P = double_pendulum([2, 1], [2, 1], [pi/2, pi/6], ode=rk4, step=0.08)
plt.plot(*P.return_energy(), label=r"rk4,$h=0.08$s")
plt.tick_params(direction='in',labelsize=20)
plt.xticks((0,10,20,30))
plt.xlabel(r"$t$(s)", fontsize=20)
plt.ylabel(r"$H$(J)", fontsize=20)
plt.legend(loc=2, frameon=False, fontsize=15)
plt.savefig("F03.pdf", bbox_inches='tight')
plt.show()
'''

P = double_pendulum([2, 1], [2, 1], [pi/2, pi/6], ode=rk4, step=0.02)
P.plot_trajectory_illustrate()