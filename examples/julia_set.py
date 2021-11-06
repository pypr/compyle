import time
from math import cos, sin
import numpy as np
from compyle.api import annotate, elementwise, get_config, wrap


@annotate
def julia(i, z, xa, ya, t):
    c0 = 0.7885*cos(t)
    c1 = 0.7885*sin(t)
    x = xa[i]
    y = ya[i]
    iters = 0
    while (x*x + y*y) < 400 and iters < 50:
        xn = x*x - y*y + c0
        y = x*y*2.0 + c1
        x = xn
        iters += 1
    z[i] = 1.0 - iters*0.02


def timer(x, y, z):
    s = time.perf_counter()
    n = 2000
    dt = 4*np.pi/n
    for i in range(n):
        julia(z, x, y, -dt*i)
    print("Took", time.perf_counter() - s, "seconds")


def plot(x, y, z, nx, ny):
    from mayavi import mlab
    mlab.figure(size=(600, 600))
    xmin, xmax = np.min(x.data), np.max(x.data)
    ymin, ymax = np.min(y.data), np.max(y.data)
    s = mlab.imshow(z.data.reshape((nx, ny)),
                    extent=[xmin, xmax, ymin, ymax, 0, 0],
                    colormap='jet')
    s.scene.z_plus_view()
    n = 2000
    dt = 4*np.pi/n
    for i in range(n):
        julia(z, x, y, -dt*i)
        z.pull()
        s.mlab_source.scalars = z.data.reshape((nx, ny))
        if i % 3 == 0:
            mlab.process_ui_events()
    mlab.show()


def save(x, y, z, gif_path='julia_set.gif'):
    import imageio as iio
    n = 250
    dt = 2*np.pi/n
    print(f"Writing {gif_path}")
    with iio.get_writer(gif_path, mode='I') as writer:
        for i in range(n):
            julia(z, x, y, -dt*i)
            z.pull()
            writer.append_data(
                (z.data.reshape((nx, ny))*255).astype(np.uint8)
            )
            print(f"{i}/{n}", end='\r')
        print("Done.  ")
    try:
        from pygifsicle import optimize
        optimize(gif_path)
    except ImportError:
        print("Install pygifsicle for an optimized GIF")


if __name__ == '__main__':
    from compyle.utils import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-n', action='store', type=int, dest='n',
                   default=512, help='Number of grid points in y.')
    p.add_argument(
        '--show', action='store_true', dest='show',
        default=False, help='Show animation (requires mayavi)'
    )
    p.add_argument(
        '--gif', action='store_true',
        default=False, help='Make a gif animation (requires imageio)'
    )
    cfg = get_config()
    cfg.suppress_warnings = True
    o = p.parse_args()
    julia = elementwise(julia)
    ny = o.n
    nx = int(4*ny//3)
    x, y = np.mgrid[-2:2:nx*1j, -1.5:1.5:ny*1j]
    x, y = x.ravel(), y.ravel()
    z = np.zeros_like(x)
    x, y, z = wrap(x, y, z)

    timer(x, y, z)

    if o.show:
        plot(x, y, z, nx, ny)
    if o.gif:
        save(x, y, z)
