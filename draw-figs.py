#!/usr/bin/env python
import sys

from os import path
import os

from data_info import *

#BASIC Options
#TITLE = 'data size v.s. ROC'
XLABEL = 'updates'
XLABEL = 'time'
#XLABEL = 'rank'
XLABEL = 'cputime'
XLABEL = 'iter'
XLABEL = 'walltime'
XLABEL = 'time'

YLABEL = r'obj'
YLABEL = 'cputime'
YLABEL = 'training-LL'
YLABEL = 'walltime'
YLABEL = r'p@10'

FILETYPE = 'png'
TITLE = '%s v.s. %s ' % (XLABEL, YLABEL)

ymin = None
ymax = None
xmax = None
xmin = None

ymax = 105
xmax = 200

COLORS = 'blue,red,black,green,cyan,magenta,chartreuse,blueviolet,forestgreen,black,aqua'.split(',')
LINESTYLES = '-,--,-.,:'.split(',')
MARKERS = 'o,s,^,v,*,+,x,.'.split(',')
mycolor = lambda i: COLORS[i%len(COLORS)]
mylinestyle = lambda i: LINESTYLES[i%len(LINESTYLES)]
mymarker = lambda i: MARKERS[i%len(MARKERS)]

logs = sys.argv[1:]

def getlegend(log):
    if 'Greedy' in log: return 'Greedy-MIPS'
    elif 'Hgreedy' in log: return 'Improved-Greedy-MIPS-heap'
    elif 'greedy' in log: return 'Greedy-MIPS'
    elif 'sample' in log: return 'Sample-MIPS'
    elif 'lsh' in log: return 'LSH-MIPS'
    elif 'pca' in log: return 'PCA-MIPS'
    elif 'diamond' in log: return 'Diamond-MSIPS'
    else: return log

def getlengend_greey_comp(log):
    if 'Greedy' in log : return 'Improved-Greedy-MIPS with Selection Tree'
    elif 'Hgreedy' in log : return 'Improved-Greedy-MIPS with Max Heap'
    elif 'greedy' in log :  return 'Original-Greedy-MIPS with Selection Tree'
    else : return log

def getcoord(log):
    x = []
    y = []
    init = []
    for line in open(log, 'r'):
        if line.strip() == "" : continue
        if XLABEL == 'rank':
            if 'oiter 1 ' not in line: continue
        line = line.split()
        if XLABEL not in line or YLABEL not in line : continue
        idx = line[::2].index(XLABEL)
        x += [line[1::2][idx]]
        idx = line[::2].index(YLABEL)
        y += [line[1::2][idx]]
    #x = map(float, x)
    x = map(float, x)
    y = map(float, y)
    x, y= zip(*sorted(zip(x,y), reverse=True))
    return [x, y, log]

def transform2reletive(curves):
    m = min(map(lambda x: min(x[1]) , curves))
    for i in range(len(curves)):
        curves[i][1] = map(lambda x: abs((x - m)/m), curves[i][1])

def scale(curves, s):
    for i in range(len(curves)):
        #curves[i][1] = [float(s)*(iter+1)/x for iter, x in enumerate(curves[i][1])]
        tmp = zip(*filter(lambda x : x[0] > 0, zip(curves[i][0], curves[i][1])))
        curves[i][0] = tmp[0]
        curves[i][1] = tmp[1]
        try :
            curves[i][0] = map(lambda x: s/(x+1e-9), curves[i][0])
        except :
            print curves[i]


def draw(curves, dataname, hline=None, naive=None, filename=None, legend=None):
    global xmax, ymax
    import matplotlib
    matplotlib.use('Agg')
    #matplotlib.rc('text',usetex=True)
    matplotlib.rc('font',family='serif')
    from matplotlib import pylab
    params = {'font.size': 18, 'axes.labelsize': 18, 'text.fontsize': 18, 'legend.fontsize': 16,'xtick.labelsize': 14,'ytick.labelsize': 14, 'axes.formatter.limits':(-3,3)}
    pylab.rcParams.update(params)

    pylab.figure()
    plots = []
    #pylab.axhline(y=hline, lw=1, c='gray', marker='.')
    for i in range(len(curves)):
        #change 'plot' to 'semilogx'/'semilogy'/'loglog' if you need it
        if dataname.lower() in ['news20', 'covtype', 'rcv1']: plotter = pylab.semilogx
        else: plotter = pylab.plot
        if 'liblinear' in curves[i][2]:
            tmp,= plotter(curves[i][0], curves[i][1],
                    lw=3, c=mycolor(4), ls=mylinestyle(4))
        else :
            tmp,= plotter(curves[i][0], curves[i][1],
                    lw=4, c=mycolor(i), ls=mylinestyle(i))
        plots += [tmp]
        #pylab.axvline(x=894956000)
    if xmax!=None: pylab.xlim(xmax=xmax)
    if ymax!=None: pylab.ylim(ymax=ymax)
    if xmin!=None: pylab.xlim(xmin=xmin)
    if ymin!=None: pylab.ylim(ymin=ymin)

    if naive: pylab.xlabel('Speedup over naive approach ({0} s)'.format(naive), fontsize='large')
    else : pylab.xlabel('Speedup over naive approach', fontsize='large')
    if YLABEL.startswith('p@'):
        pylab.ylabel('Performance (prec@{0})'.format(YLABEL.split('@')[-1]), fontsize='large')
    elif YLABEL.startswith('n@'):
        pylab.ylabel('Performance (nDCG@{0})'.format(YLABEL.split('@')[-1]), fontsize='large')
    else :
        pylab.ylabel('Performance ({0})'.format(YLABEL), fontsize='large')
    title = dataname
    tmpm = '17,770' if 'netflix' in dataname else '624,961'
    if 'pos' in dataname or 'syn' in dataname:
        m = int(dataname.split('.')[1][1:])
        d = int(dataname.split('.')[2][1:])
        if 'pos' in dataname:
            title = 'syn-uniform ($n=2^{%d}, k=2^{%d}$)'%(m,d)
        else :
            title = 'syn-normal ($n=2^{%d}, k=2^{%d}$)'%(m,d)
    elif dataname in ['netflix','yahoo']:
        title = '%s ($n=%s, k=100$)'%(dataname,tmpm)
    elif dataname == 'netflix50':
        title = '%s ($n=%s, k=50$)'%('netflix',tmpm)
    elif dataname == 'netflix200':
        title = '%s ($n=%s, k=200$)'%('netflix',tmpm)
    elif dataname == 'yahoo50':
        title = '%s ($n=%s, k=50$)'%('yahoo',tmpm)
    elif dataname == 'yahoo200':
        title = '%s ($n=%s, k=200$)'%('yahoo',tmpm)
    pylab.title('%s'%(title), fontsize='large')
    if legend == None: legend = getlegend
    pylab.legend(plots, map(lambda x: legend(x[2]), curves), loc='best')
    if filename:
        pylab.savefig('figs/%s-%s-%s.%s'%(dataname.lower(), filename, YLABEL, FILETYPE), format=FILETYPE)
    else :
        pylab.savefig('figs/%s-comp-%s.%s'%(dataname.lower(), YLABEL, FILETYPE), format=FILETYPE)

def get_naive(data):
    return float(open('logs/{0}-naive.raw'.format(data)).read().split()[3])

solver = ['greedy', 'pca', 'lsh', 'sample', 'Greedy', 'Hgreedy']

def go_one(data, solver, filename=None, legend=None):
    logs = []
    for s in solver:
        if 'pos' not in data and s == 'sample':
            s = 'diamond'
        logs += ['logs/%s-%s.raw' %(data,s)]
    print logs
    curves = map(getcoord, logs)
    naive = get_naive(data)
    scale(curves, naive)
    draw(curves, data, naive=naive, filename=filename, legend=legend)


if not path.exists('figs'):
    os.system('mkdir -p figs')

solver = ['Greedy',  'pca', 'lsh', 'sample']
for label in ['p@1', 'n@1', 'p@5', 'n@5', 'p@10', 'n@10'] :
    YLABEL = label
    for d in datasets:
        xmax = 200
        if d in synthetic_set:
            if d.startswith('pos'):
                xmax = 60
            else:
                xmax = 150
        go_one(d, solver)

