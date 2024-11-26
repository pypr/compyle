import numpy as np
from compyle.api import annotate, Elementwise, get_config, declare
from compyle.autodiff_tapenade import ReverseGrad
import taichi as ti

from time import time

# This problem aims at solving the billiards problem using autodiff. 
# Aim is to determine initial location and velocity of a cue ball to hit the 
# target location with one of the decided balls on the table.

get_config().use_openmp = True
gui = ti.GUI("Billiards", (1024, 1024), background_color=0x3C733F)

def visualise(x, y, n_balls, goalx, goaly, pixel_radius, steps):
    gui.clear()
    for t in range(1, steps):
        gui.circle((goalx, goaly), 0x00000, pixel_radius // 2)
        for i in range(n_balls):
            idxi = t * n_balls + i
            if i == 0:
                color = 0xCCCCCC
            elif i == n_balls - 1:
                color = 0x3344cc
            else:
                color = 0xF20530

            gui.circle((x[idxi], y[idxi]), color, pixel_radius)

        gui.show()

# forward kernel to simulate billiards for given initial conditions and steps
@annotate(int='n_balls, target_ball, billiard_layers, steps',
          float='dt, elasticity, goalx, goaly, radius',
          floatp='x, y, vx, vy, init_x, init_y, init_vx, init_vy, impulse_x, impulse_y, x_inc, y_inc, loss')
def forward(x, y, vx, vy, init_x, init_y, init_vx, init_vy,
            impulse_x, impulse_y, x_inc, y_inc, n_balls, dt,
            elasticity, target_ball, goalx, goaly, loss,
            billiard_layers, radius, steps):
    #initialize
    x[0] = init_x[0]
    y[0] = init_y[0]
    vx[0] = init_vx[0]
    vy[0] = init_vy[0]
    count = declare('int')
    idxi = declare('int')
    idxj = declare('int')
    idxip = declare('int')
    idxtg = declare('int')
    i, j, t = declare('int')
    count = 0
    for i in range(billiard_layers):
        for j in range(i + 1):
            count += 1
            x[count] = i * 2 * radius + 0.5
            y[count] = j * 2 * radius + 0.5 - i * radius * 0.7
    
    for t in range(1, steps):
        # collide balls
        for i in range(n_balls):
            x_inc[i] = 0
            y_inc[i] = 0
            impulse_x[i] = 0
            impulse_y[i] = 0
            for j in range(n_balls):
                if i != j:
                    x_inc_contrib = 0.0
                    y_inc_contrib = 0.0
                    impx = 0
                    impy = 0
                    
                    idxi = (t - 1) * n_balls + i
                    idxj = (t - 1) * n_balls + j
                    distx = (x[idxi] + dt * vx[idxi]) - (x[idxj] + dt * vx[idxj])
                    disty = (y[idxi] + dt * vy[idxi]) - (y[idxj] + dt * vy[idxj])
                    dist_norm = ((distx * distx) + (disty * disty)) ** 0.5
                    rela_vx = vx[idxi] - vx[idxj]
                    rela_vy = vy[idxi] - vy[idxj]
                    
                    if dist_norm < 2 * radius:
                        dirx = distx / dist_norm
                        diry = disty / dist_norm
                        projected_v = (dirx * rela_vx) + (diry * rela_vy)

                        if projected_v < 0:
                            impx = -(1 + elasticity) * 0.5 * projected_v * dirx
                            impy = -(1 + elasticity) * 0.5 * projected_v * diry
                            
                            toi = (dist_norm - 2 * radius) / min(
                                -1e-3, projected_v)  # Time of impact
                            x_inc_contrib = min(toi - dt, 0.0) * impx
                            y_inc_contrib = min(toi - dt, 0.0) * impy
                            
                    x_inc[i] += x_inc_contrib
                    y_inc[i] += y_inc_contrib
                    impulse_x[i] += impx
                    impulse_y[i] += impy

        # end collide balls

        # update speed and position
        for i in range(n_balls):
            idxi = t * n_balls + i
            idxip = (t - 1) * n_balls + i
            vx[idxi] = vx[idxip] + impulse_x[i]
            vy[idxi] = vy[idxip] + impulse_y[i]
            x[idxi] = x[idxip] + dt * vx[idxi] + x_inc[i]
            y[idxi] = y[idxip] + dt * vy[idxi] + y_inc[i]
            
    # compute loss
    idxtg = (steps - 1) * n_balls + target_ball
    loss[0] = (x[idxtg] - goalx) ** 2 + (y[idxtg] - goaly) ** 2


# generate a gradient function for the forward function
grad_forward = ReverseGrad(forward, wrt=['init_x', 'init_y', 'init_vx', 'init_vy', 'y_inc', 'x', 'y', 'vx', 'x_inc', 'vy', 'impulse_x', 'impulse_y'], gradof=['loss'])


def optimize():    
    for iter in range(200):
        lossb[0] = 1.0
        
        grad_forward(x, xb, y, yb, vx, vxb, vy, vyb, init_x, init_xb, init_y, init_yb,
                     init_vx, init_vxb, init_vy, init_vyb, impulse_x, impulse_xb,
                     impulse_y, impulse_yb, x_inc, x_incb, y_inc, y_incb,
                     n_balls, dt, elasticity, target_ball, goalx, goaly, loss, lossb,
                     billiard_layers, radius, steps)
        init_x[0] -= learning_rate * init_xb[0]
        init_y[0] -= learning_rate * init_yb[0]
        init_vx[0] -= learning_rate * init_vxb[0]
        init_vy[0] -= learning_rate * init_vyb[0]
        if iter % 20 == 0:
            print(f"iter: {iter} \t loss: {loss[0]}")
if __name__ == '__main__':
    # setup parameters
    dtype = np.float32
    billiard_layers = 4
    n_balls = 1 + (1 + billiard_layers) * billiard_layers // 2

    vis_interval = 64
    output_vis_interval = 16
    steps = 1024
    max_steps = 1024

    vis_resolution = 1024

    loss = np.zeros(1, dtype=dtype)

    x = np.zeros(max_steps * n_balls, dtype=dtype)
    y = np.zeros(max_steps * n_balls, dtype=dtype)
    x_inc = np.zeros(n_balls, dtype=dtype)
    y_inc = np.zeros(n_balls, dtype=dtype)
    vx = np.zeros(max_steps * n_balls, dtype=dtype)
    vy = np.zeros(max_steps * n_balls, dtype=dtype)
    impulse_x = np.zeros(n_balls, dtype=dtype)
    impulse_y = np.zeros(n_balls, dtype=dtype)

    init_x = np.array([0.1], dtype=dtype)
    init_y = np.array([0.5], dtype=dtype)
    init_vx = np.array([0.3], dtype=dtype)
    init_vy = np.array([0.0], dtype=dtype)
    
    xb = np.zeros_like(x, dtype=dtype)
    yb = np.zeros_like(y, dtype=dtype)
    vxb = np.zeros_like(vx, dtype=dtype)
    vyb = np.zeros_like(vy, dtype=dtype)
    init_xb = np.zeros_like(init_x, dtype=dtype)
    init_yb = np.zeros_like(init_y, dtype=dtype)
    init_vxb = np.zeros_like(init_vx, dtype=dtype)
    init_vyb = np.zeros_like(init_vy, dtype=dtype)
    impulse_xb = np.zeros_like(impulse_x, dtype=dtype)
    impulse_yb = np.zeros_like(impulse_y, dtype=dtype)
    x_incb = np.zeros_like(x_inc, dtype=dtype)
    y_incb = np.zeros_like(y_inc, dtype=dtype)
    lossb = np.ones_like(loss, dtype=dtype)

    target_ball = n_balls - 1
    goalx = 0.9
    goaly = 0.75
    radius = 0.03
    elasticity = 0.8


    dt = 0.003
    learning_rate = 0.01
    
    begin = time()
    optimize()
    end = time()
    print(f"took: {end - begin} seconds to simulate {billiard_layers} layers of billiard balls")
    
    pixel_radius = (int(radius * 1024) + 1)
    visualise(x, y, n_balls, goalx, goaly, pixel_radius, steps)