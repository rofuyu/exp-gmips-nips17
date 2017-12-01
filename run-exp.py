#!/usr/bin/env python


import sys
from os import path
from commands import getoutput

dryrun = False
datadir = 'data/'
logdir = 'logs/'

if not path.exists('./src/go'):
    getoutput('make -C src')

from data_info import *

def run_greedy():
    cmd_template = './src/go {datadir}/{data}.tc greedy {options} > {logdir}/{log}.raw'
    for data in datasets:
        if data in real_set:
            m = real_m[data]
            d = real_d[data]
        else :
            m = int(data.split('.')[1][1:])
            d = int(data.split('.')[2][1:])
        options = ' '.join(str(2**budget) for budget in range(d,m+1))
        log = '{data}-greedy'.format(data=data)
        cmd = cmd_template.format(data=data,options=options,log=log,datadir=datadir,logdir=logdir)
        if path.exists('{logdir}/{log}.raw'.format(logdir=logdir,log=log)): continue
        print(cmd)
        if not dryrun: getoutput(cmd)

def run_improved_greedy():
    cmd_template = './src/go {datadir}/{data}.tc Greedy-improve {options} > {logdir}/{log}.raw'
    for data in datasets:
        if data in real_set:
            m = real_m[data]
            d = real_d[data]
        else :
            m = int(data.split('.')[1][1:])
            d = int(data.split('.')[2][1:])
        options = ' '.join(str(2**budget) for budget in range(d,m+1))
        log = '{data}-Greedy'.format(data=data)
        cmd = cmd_template.format(data=data,options=options,log=log,datadir=datadir,logdir=logdir)
        if path.exists('{logdir}/{log}.raw'.format(logdir=logdir,log=log)): continue
        print(cmd)
        if not dryrun: getoutput(cmd)

def run_improved_heap_greedy():
    cmd_template = './src/go {datadir}/{data}.tc Hgreedy-improve {options} > {logdir}/{log}.raw'
    for data in datasets:
        if data in real_set:
            m = real_m[data]
            d = real_d[data]
        else :
            m = int(data.split('.')[1][1:])
            d = int(data.split('.')[2][1:])
        options = ' '.join(str(2**budget) for budget in range(d-2,m-1))
        log = '{data}-Hgreedy'.format(data=data)
        cmd = cmd_template.format(data=data,options=options,log=log,datadir=datadir,logdir=logdir)
        if path.exists('{logdir}/{log}.raw'.format(logdir=logdir,log=log)): continue
        print(cmd)
        if not dryrun: getoutput(cmd)

def run_diamond():
    cmd_template = './src/go {datadir}/{data}.tc diamond {options} > {logdir}/{log}.raw'
    for data in datasets:
        if data in real_set:
            m = real_m[data]
            d = real_d[data]
        else :
            m = int(data.split('.')[1][1:])
            d = int(data.split('.')[2][1:])
        options = ' '.join(str(2**budget) for budget in range(d,m+1))
        log = '{data}-diamond'.format(data=data)
        cmd = cmd_template.format(data=data,options=options,log=log,datadir=datadir,logdir=logdir)
        if path.exists('{logdir}/{log}.raw'.format(logdir=logdir,log=log)): continue
        print(cmd)
        if not dryrun: getoutput(cmd)

def run_lsh():
    cmd_template = './src/go {datadir}/{data}.tc lsh {options} > {logdir}/{log}.raw'
    for data in datasets:
        if data in real_set:
            m = real_m[data]
            d = real_d[data]
        else :
            m = int(data.split('.')[1][1:])
            d = int(data.split('.')[2][1:])
        options = []
        for B in [20, 40, 80, 160]:
            for R in [5, 8, 11, 14, 17, 20]:
                options += ['{0} {1}'.format(B,R)]
        options = ' '.join(options)
        log = '{data}-lsh'.format(data=data)
        if path.exists('{logdir}/{log}.raw'.format(logdir=logdir,log=log)): continue
        cmd = cmd_template.format(data=data,options=options,log=log,datadir=datadir,logdir=logdir)
        print(cmd)
        if not dryrun: getoutput(cmd)

def run_pca():
    cmd_template = './src/go {datadir}/{data}.tc pca {options} > {logdir}/{log}.raw'
    for data in datasets:
        if data in real_set:
            m = real_m[data]
            d = real_d[data]
        else :
            m = int(data.split('.')[1][1:])
            d = int(data.split('.')[2][1:])
        if d == 2:
            options = ' 1 2 3 4'
        else :
            options = ' '.join(str(depth) for depth in range(2,12) if depth <= 2**d)
        log = '{data}-pca'.format(data=data)
        if path.exists('{logdir}/{log}.raw'.format(logdir=logdir,log=log)): continue
        cmd = cmd_template.format(data=data,options=options,log=log,datadir=datadir,logdir=logdir)
        print(cmd)
        if not dryrun: getoutput(cmd)

def run_naive():
    cmd_template = './src/go {datadir}/{data}.tc naive {options} > {logdir}/{log}.raw'
    for data in datasets:
        options = ''
        log = '{data}-naive'.format(data=data)
        if path.exists('{logdir}/{log}.raw'.format(logdir=logdir,log=log)): continue
        cmd = cmd_template.format(data=data,options=options,log=log,datadir=datadir,logdir=logdir)
        print(cmd)
        if not dryrun: getoutput(cmd)

def run_all():
    run_naive()
    run_improved_greedy()
    run_diamond()
    run_lsh()
    run_pca()

def main(argv):
    solvers = {
            'all': run_all,
            'naive': run_naive,
            'greedy': run_improved_greedy,
            'diamond' : run_diamond,
            'lsh' : run_lsh,
            'pca' : run_pca,
            #'greedy-with-selection': run_greedy,
            #'improved-greedy-with-selection': run_improved_greedy,
            #'improved-greedy-with-heap': run_improved_heap_greedy,
            }
    if len(argv) != 2 :
        print('{0} algo'.format(argv[0]))
        print('  algo: {}'.format('|'.join(sorted(solvers.keys()))))
        return
    if path.exists(datadir) == False:
        print('datadir {0} does not exist!'.format(datadir))
        return
    if path.exists(logdir) == False:
        getoutput('mkdir -p {0}'.format(logdir))
    if argv[1] in solvers:
        solvers[argv[1]]()
    else:
        raise KeyError('{} is not one of {}'.format(argv[1], '|'.join(solvers.keys())))

if __name__ == '__main__':
    main(sys.argv)
