from os import path

real_set = ['netflix50', 'netflix200', 'yahoo50', 'yahoo200']

# real_m should be roughly equal to the number of log2 (the number of candiates in each real data set)
real_m = {
    'netflix50': 15,
    'netflix200': 15,
    'yahoo50': 20,
    'yahoo200': 20
    }

# real_m should be roughtly equal to the number of log2 (the dimension of the embeddings)
real_d={
    'netflix50': 5,
    'netflix200': 7,
    'yahoo50': 5,
    'yahoo200': 7
    }

synthetic_set = set()

synthetic_set |= set('syn.m18.d{0}'.format(d) for d in [2,5,7,10])
synthetic_set |= set('syn.m{0}.d7'.format(m) for m in [17,18,19,20])
synthetic_set |= set('pos.m18.d{0}'.format(d) for d in [2,5,7,10])
synthetic_set |= set('pos.m{0}.d7'.format(m) for m in [17,18,19,20])
datasets = real_set + list(synthetic_set)

datasets = list(filter(lambda x: path.exists('data/{}.tc'.format(x)), datasets))

