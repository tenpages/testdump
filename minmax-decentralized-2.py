import numpy as np
import random

# suppose the network is a-b-c (a and c are not connected directly)

x = [0,0,0,0,0]
x[0] = random.random() * 8 - 4
x[1] = random.random() * 8 - 4
x[2] = random.random() * 8 - 4
x[3] = random.random() * 8 - 4
x[4] = random.random() * 8 - 4

# y[0] = .5(x-2)^2
# y[1] = .5(x+2)^2
# y[2] = .1x^2+5
# y[3] = .3(x-3)^2+2
# y[4] = .05(x+1)^2+5.5

# connections = [[1,1,0,0,0],
# [1,1,1,0,0],
# [0,1,1,0,0],
# [1,0,0,1,0],
# [0,1,1,0,1]]
connections = [[1,2,3,4,1],
[2,1,5,2,5],
[3,5,1,3,2],
[4,2,3,1,1],
[1,5,2,1,1]]

xs = [[x[0], x[1], x[2], x[3], x[4]]]
ys = []
nabla_ys = []

nabla_y = [0,0,0,0,0]
y = [0,0,0,0,0]

for k in range(10000):
    nabla_y[0] = x[0] - 2
    nabla_y[1] = x[1] + 2
    nabla_y[2] = 0.2 * x[2]
    nabla_y[3] = 0.6 * x[3] - 1.8
    nabla_y[4] = 0.1 * x[4] + 0.1

    y[0] = .5 * (x[0] - 2) ** 2
    y[1] = .5 * (x[1] + 2) ** 2
    y[2] = .1 * x[2] ** 2 + 5
    y[3] = .3 * (x[3] - 3) ** 2 + 2
    y[4] = .05 * (x[4] + 1) ** 2 + 5.5

    # print("{}\t{}\t{}".format(y[0], y[1], y[2], y[3], y[4]))
    # print("{}\t{}\t{}".format(nabla_y[0], nabla_y[1], nabla_y[2], nabla_y[3], nabla_y[4]))
    # print()

    # eta = 1 / (i + 1)
    eta = .01
    z = []
    g_theta = []
    for i in range(5):
        z_i = 0
        count = 0
        max_y_i = -1000000
        max_y_i_i = i
        for j in range(5):
            if k % connections[i][j] == 0:
                z_i = z_i + xs[-1][j]
                count += 1
                if max_y_i < y[j]:
                    max_y_i = y[j]
                    max_y_i_i = j
        z_i = z_i / count
        z.append(z_i)
        g_theta.append(nabla_y[max_y_i_i])
        x[i] = z[i] - eta * g_theta[i]

    xs.append(x.copy())
    ys.append(y.copy())
    nabla_ys.append(nabla_y.copy())

print(xs[-1])
print(ys[-1])
print(nabla_ys[-1])

import matplotlib as mpl
import csv
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.labelweight'] = "bold"
mpl.rcParams['font.weight']= "bold"
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
mpl.rcParams['xtick.major.width'] = 0.3
mpl.rcParams['xtick.major.size'] = 1
mpl.rcParams['xtick.major.pad'] = .5
mpl.rcParams['xtick.labelsize']= 3
mpl.rcParams['ytick.major.width'] = 0.3
mpl.rcParams['ytick.major.size'] = 1
mpl.rcParams['ytick.major.pad'] = .5
mpl.rcParams['ytick.labelsize']= 3
mpl.rcParams['figure.subplot.wspace']=0.20
mpl.rcParams['figure.subplot.hspace']=0.09
mpl.rcParams['savefig.bbox']='tight'
mpl.rcParams['savefig.pad_inches']=0.03
mpl.rcParams['legend.fontsize']=3
mpl.rcParams['legend.frameon']=True
mpl.rcParams['legend.framealpha']=1
mpl.rcParams['legend.borderaxespad']=.5
mpl.rcParams['legend.fancybox']=False
mpl.rcParams['grid.linewidth']=0.3
# mpl.rcParams['line.linewidth']=0.3
mpl.rcParams['hatch.linewidth']=0.3

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(1.6,1.4))
xs = np.array(xs).transpose()

# plt.xticks()
plt.xlabel("iteration $t$",{'size':3},labelpad=-.3)
plt.ylabel("value of local estimates $x$",{'size':3},labelpad=-.3)

plots = []
plots.append(plt.plot(range(10001), xs[0], ls='-', label="agent 1", linewidth=.3, marker='^', markevery=1000, ms=.6))
plots.append(plt.plot(range(10001), xs[1], ls='-', label="agent 2", linewidth=.3, marker='o', markevery=1000, ms=.6))
plots.append(plt.plot(range(10001), xs[2], ls='-', label="agent 3", linewidth=.3, marker='v', markevery=1000, ms=.6))
plots.append(plt.plot(range(10001), xs[3], ls='-', label="agent 3", linewidth=.3, marker='P', markevery=1000, ms=.6))
plots.append(plt.plot(range(10001), xs[4], ls='-', label="agent 3", linewidth=.3, marker='s', markevery=1000, ms=.6))

legend_names = ['agent 1', 'agent 2', 'agent 3', 'agent 4', 'agent 5']
legend = plt.legend(plots, labels = legend_names, loc="lower right")#, bbox_to_anchor=(1,.8))

plt.savefig("tests-comm-freq.pdf")