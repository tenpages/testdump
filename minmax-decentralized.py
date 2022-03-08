import numpy as np
import random

# suppose the network is a-b-c (a and c are not connected directly)

x_a = random.random() * 8 - 4
x_b = random.random() * 8 - 4
x_c = random.random() * 8 - 4

# y_a = .5(x-2)^2
# y_b = .5(x+2)^2
# y_c = .1x^2+5

xs = [[x_a, x_b, x_c]]
ys = []
nabla_ys = []

for i in range(1000):
    nabla_y_a = x_a - 2
    nabla_y_b = x_b + 2
    nabla_y_c = 0.2 * x_c
    y_a = .5 * (x_a - 2) ** 2
    y_b = .5 * (x_b + 2) ** 2
    y_c = .1 * x_c ** 2 + 5
    print("{}\t{}\t{}".format(y_a, y_b, y_c))
    print("{}\t{}\t{}".format(nabla_y_a, nabla_y_b, nabla_y_c))
    print()
    eta = 1 / (i + 1)
    z_a = (x_a + x_b) / 2
    z_b = (x_a + x_b + x_c) / 3
    z_c = (x_b + x_c) / 2
    g_theta_a = [nabla_y_a, nabla_y_b][np.argmax((y_a, y_b))]
    g_theta_b = [nabla_y_a, nabla_y_b, nabla_y_c][np.argmax((y_a, y_b, y_c))]
    g_theta_c = [nabla_y_b, nabla_y_c][np.argmax((y_b, y_c))]
    x_a = z_a - eta * g_theta_a
    x_b = z_b - eta * g_theta_b
    x_c = z_c - eta * g_theta_c
    xs.append([x_a, x_b, x_c])
    ys.append([y_a, y_b, y_c])
    nabla_ys.append([nabla_y_a, nabla_y_b, nabla_y_c])

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
mpl.rcParams['line.linewidth']=0.3
mpl.rcParams['hatch.linewidth']=0.3

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(1.1,0.8))
xs = np.array(xs).transpose()

# plt.xticks()
plt.xlabel("iteration $t$",{'size':3},labelpad=-.3)
plt.ylabel("value of local estimates $x$",{'size':3},labelpad=-.3)

plots = []
plots.append(plt.plot(range(1001), xs[0], ls='-', label="agent 1", linewidth=.3, marker='^', markevery=100, ms=.6))
plots.append(plt.plot(range(1001), xs[1], ls='-', label="agent 2", linewidth=.3, marker='o', markevery=100, ms=.6))
plots.append(plt.plot(range(1001), xs[2], ls='-', label="agent 3", linewidth=.3, marker='v', markevery=100, ms=.6))

legend_names = ['agent 1', 'agent 2', 'agent 3']
legend = plt.legend(plots, labels = legend_names, loc="upper center")#, bbox_to_anchor=(1,.8))

plt.savefig("tests.pdf")