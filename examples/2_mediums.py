import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

if __name__ == '__main__':
    xmin = -2.5
    xmax = 2.5
    xstep = .05
    x = np.arange(xmin, xmax+xstep, xstep)

    xt, zt = 1, 0
    xf, zf = 1, 5

    z1 = np.power(x, 2)/5 + 3

    c1 = 1.49
    c2 = 5.3

    k = parrilla_2007_generalized(
        surface=(x, z1),
        focus=(xf, zf),
        transmitter=(xt, zt),
        c1=c1,
        c2=c2
    )




    plt.figure()
    plt.plot(xt, zt, 'sk')
    plt.plot(x, z1, 'o-r')
    plt.plot(x[k], z1[k], 'ob')
    plt.plot(xf, zf, 'xk')
    plt.show()
