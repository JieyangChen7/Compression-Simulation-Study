import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

NumNodes = ['1', '2', '4', '8', '16', '32', '64', '128']
WriteTypes = ['Write', 
              'Write + MGARD', 
              'Write + MGARD w/ prefetch']
ReadTypes = ['Read', 
            'Read + MGARD', 
            'Read + MGARD w/ prefetch']
Color = ['teal', 'olivedrab', 'lawngreen', 'yellowgreen', 'maroon', 'lightcoral', 'maroon', 'lightcoral', 'yellowgreen', 'lightcoral', 'teal']

# Summit: 1 GPUs 1 writers 1 node
#EB=1e17
ngpus = 1;
var_size = 1.59595
data_size1_1 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time1_1 = [
      [4.82741, 6.27101, 9.12839, 11.8553, 14.4842, 20.6159], # no compression write
      [0.62418, 1.31698, 2.68654, 3.51258, 4.40481, 5.42988], # no decompression read
      [0.262716, 0.464392, 0.828489, 1.27163, 1.60339, 2.14141], # no prefetch compress
      [0.251253, 0.460072, 0.80947, 1.24023, 1.56028, 2.1525], # no prefetch decompress
      [0.244444, 0.310902, 0.550307, 0.671181, 0.85735, 1.01797], # prefetch compress
      [0.243257, 0.460687, 0.543815, 0.684399, 0.847096, 1.06177], # prefetch decompress
      [0.281609, 0.311321, 0.415496, 0.482469, 0.583175, 0.863933], # write
      [0.0845692, 0.0918932, 0.112745, 0.119458, 0.140515, 0.174732], # read
]

# Summit: 2 GPUs 2 writers 1 node
#EB=1e17
ngpus = 2;
var_size = 1.59595
data_size1_2 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time1_2 = [
      [4.96244, 6.53587, 9.37912, 12.405, 15.421, 21.7765], # no compression write
      [0.857865, 1.68382, 3.45667, 5.0581, 6.70392, 8.32517], # no decompression read
      [0.376471, 0.682173, 1.13085, 1.70964, 2.19491, 2.82644], # no prefetch compress
      [0.343888, 0.651896, 1.10042, 1.64437, 2.13563, 2.84515], # no prefetch decompress
      [0.350946, 0.673538, 0.773191, 0.808153, 1.03913, 1.11551], # prefetch compress
      [0.340583, 0.653145, 0.743611, 0.835931, 1.05228, 1.19348], # prefetch decompress
      [0.28806, 0.319505, 0.425286, 0.499504, 0.613808, 0.902123], # write
      [0.0812342, 0.0853363, 0.110461, 0.141373, 0.196755, 0.236975], # read
]


# Summit: 4 GPUs 4 writers 1 node
#EB=1e17
ngpus = 4;
var_size = 1.59595
data_size1_4 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time1_4 = [
      [5.10923, 6.55698, 9.39139, 12.4424, 15.4413, 21.8453], # no compression write
      [1.0894, 1.92979, 4.09209, 5.58415, 7.3661, 32.3591], # no decompression read
      [0.609778, 1.12372, 1.64509, 2.56383, 3.33173, 4.03574], # no prefetch compress
      [0.55665, 1.03322, 1.59014, 2.25428, 2.94427, 3.87124], # no prefetch decompress
      [0.589592, 1.09029, 1.09257, 1.11546, 1.44327, 1.37864], # prefetch compress
      [0.544538, 1.03398, 1.13458, 1.12309, 1.44283, 1.44171], # prefetch decompress
      [0.294773, 0.329446, 0.426015, 0.50012, 0.614669, 0.904525], # write
      [0.0961815, 0.0939741, 0.107291, 0.149583, 0.200829, 0.401812], # read
]

# Summit: 6 GPUs 6 writers 1 node
#EB=1e17
ngpus = 6;
var_size = 1.59595
data_size1 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time1 = [
      [5.11195, 6.71289, 9.44909, 12.7746, 16.5384, 24.0889], # no compression write
      [1.79481, 3.36223, 4.683, 7.95644, 10.298, 31.6762], # no decompression read
      [0.861731, 1.47055, 2.24086, 3.33206, 4.24243, 5.67175], # no prefetch compress
      [0.732247, 1.43402, 2.2824, 3.32492, 4.39721, 5.59626], # no prefetch decompress
      [0.875847, 1.52647, 1.53246, 1.40829, 1.84573, 1.6611], # prefetch compress
      [0.758039, 1.43469, 1.53498, 1.43361, 1.89269, 1.68502], # prefetch decompress
      [0.026272, 0.0500519, 0.168013, 0.251549, 0.35717, 0.418346], # write
      [0.0618431, 0.0794002, 0.140833, 0.234959, 0.283427, 0.275598], # read
]

# Summit: 12 GPUs 12 writers 2 node
#EB=1e17
ngpus = 12;
var_size = 1.59595
data_size2 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time2 = [
      [5.08795, 6.82029, 9.50008, 12.7833, 16.7455, 24.0993], # no compression write
      [1.84607, 3.55515, 5.82714, 7.8628, 10.695, 30.7549], # no decompression read
      [0.834912, 1.6229, 2.24727, 3.49108, 4.39688, 5.74464], # no prefetch compress
      [0.770847, 1.43442, 2.27382, 3.33497, 4.40271, 5.51861], # no prefetch decompress
      [0.746699, 1.51912, 1.54477, 1.50117, 1.84775, 1.67478, ], # prefetch compress
      [0.717993, 1.43366, 1.5299, 1.43751, 1.8855, 1.68344], # prefetch decompress
      [0.0273998, 0.0551891, 0.193293, 0.270666, 0.392129, 0.483752], # write
      [0.0680841, 0.073294, 0.127009, 0.189096, 0.275726, 0.309784], # read
]

# Summit: 24 GPUs 24 writers 4 node
#EB=1e17
ngpus = 24;
var_size = 1.59595
data_size4 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time4 = [
      [5.08549, 6.76639, 9.48192, 12.78, 16.5495, 24.1107], # no compression write
      [1.62951, 3.63237, 6.53797, 9.99089, 10.873, 30.2483], # no decompression read
      [0.840935, 1.57946, 2.58943, 3.46152, 4.51392, 5.60668, ], # no prefetch compress
      [0.780244, 1.46332, 2.29419, 3.32688, 4.33172, 5.57174], # no prefetch decompress
      [0.788244, 1.50856, 1.50423, 1.43419, 1.86773, 1.62272], # prefetch compress
      [0.738338, 1.42549, 1.55576, 1.43322, 1.86121, 1.70345], # prefetch decompress
      [0.295528, 0.330522, 0.454781, 0.527427, 0.679323, 0.982912], # write
      [0.0683882, 0.093314, 0.164049, 0.229376, 0.311458, 0.388978], # read
]


# Summit: 48 GPUs 48 writers 8 node
#EB=1e17
ngpus = 48
var_size = 1.59595
data_size8 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time8 = [
      [5.10812, 6.85797, 10.1514, 12.9809, 16.6922, 24.1422], # no compression write
      [2.02745, 3.81629, 7.35429, 10.8163, 11.0501, 31.3039], # no decompression read
      [0.875292, 1.52937, 2.62297, 3.43726, 4.5335, 5.78157], # no prefetch compress
      [0.799914, 1.45356, 2.28594, 3.32021, 4.36591, 5.64887], # no prefetch decompress
      [0.810621, 1.53972, 1.49704, 1.40778, 1.95618, 1.59907], # prefetch compress
      [0.749356, 1.42894, 1.53225, 1.43545, 1.87078, 1.70266], # prefetch decompress
      [0.295552, 0.331103, 0.457265, 0.541753, 0.674424, 0.981787], # write
      [0.113215, 0.117397, 0.283734, 0.230475, 0.665491, 0.386516], # read
]

# Summit: 96 GPUs 96 writers 16 node
#EB=1e17
ngpus = 96
var_size = 1.59595
data_size16 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time16 = [
      [5.10901, 6.82416, 9.46531, 12.7783, 16.5375, 25.5404], # no compression write
      [2.10835, 4.52833, 7.50981, 10.941, 17.6937, 34.051], # no decompression read
      [1.12672, 1.53136, 2.65342, 3.56641, 4.52251, 5.71247], # no prefetch compress
      [0.825455, 1.42603, 2.27559, 3.35633, 4.36387, 5.64775], # no prefetch decompress
      [0.827463, 1.51289, 1.50293, 1.42939, 1.94145, 1.62359], # prefetch compress
      [0.771322, 1.45627, 1.56952, 1.45656, 1.87326, 1.72993], # prefetch decompress
      [0.296692, 0.331493, 0.458577, 0.541079, 0.674437, 0.982868], # write
      [0.119195, 0.0932945, 0.169349, 0.229672, 0.52229, 0.387392], # read
]

# Summit: 192 GPUs 192 writers 32 node
#EB=1e17
ngpus = 192
var_size = 1.59595
data_size32 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time32 = [
      [5.11394, 6.81565, 9.46656, 12.785, 16.5427, 24.1027], # no compression write
      [2.19867, 5.70942, 7.7329, 12.041, 14.0021, 35.9696], # no decompression read
      [1.16525, 1.53998, 2.71088, 3.49545, 4.53654, 5.80036], # no prefetch compress
      [0.819677, 1.46046, 2.31767, 3.36363, 4.38286, 5.74417], # no prefetch decompress
      [0.803399, 1.59419, 1.55658, 1.4128, 1.93768, 1.69125], # prefetch compress
      [0.769844, 1.46262, 1.58555, 1.46239, 1.8821, 1.84975], # prefetch decompress
      [0.29565, 0.331094, 0.458528, 0.544198, 0.670812, 0.982585, ], # write
      [0.140203, 0.0795893, 0.169338, 0.228692, 0.314054, 0.387455], # read
]

# Summit: 384 GPUs 384 writers 64 node
#EB=1e17
ngpus = 384
var_size = 1.59595
data_size64 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time64 = [
      [5.10389, 6.84531, 10.012, 12.7872, 16.5632, 24.1325], # no compression write
      [3.27337, 5.56436, 9.17206, 11.1188, 18.197, 35.7173], # no decompression read
      [1.14126, 1.69348, 2.74661, 3.46934, 4.72636, 5.80563], # no prefetch compress
      [0.826541, 1.47126, 2.33018, 3.35676, 4.3944, 5.70673], # no prefetch decompress
      [0.925186, 1.5618, 1.68028, 1.40886, 1.83414, 1.72526], # prefetch compress
      [0.831497, 1.46232, 1.58894, 1.49445, 1.94272, 1.80548], # prefetch decompress
      [0.297117, 0.332595, 0.45859, 0.54623, 0.674684, 0.983264], # write
      [0.163115, 0.162913, 0.172913, 0.213048, 0.313048, 0.389272], # read
]

# Summit: 768 GPUs 768 writers 128 node
#EB=1e17
ngpus = 768
var_size = 1.59595
data_size128 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time128 = [
      [5.12255, 6.83743, 9.98949, 12.7848, 16.5537, 28.6877], # no compression write
      [3.28943, 6.38299, 8.98806, 12.3869, 31.4023, 50.1854], # no decompression read
      [1.131, 1.5984, 2.77674, 3.52493, 4.67682, 5.77807], # no prefetch compress
      [0.829916, 1.59685, 2.32322, 3.3828, 4.45448, 5.63969], # no prefetch decompress
      [0.91734, 1.55295, 1.51733, 1.40949, 1.83303, 1.72526], # prefetch compress
      [0.824813, 1.49475, 1.5831, 1.57817, 1.8774, 1.80548], # prefetch decompress
      [0.297496, 0.331717, 0.458567, 0.545241, 0.67185, 0.977459], # write
      [0.503431, 1.7511, 5.26833, 3.37943, 2.31987, 4.80666], # read
]

# Summit: 1536 GPUs 1536 writers 128 node
#EB=1e17
ngpus = 1536
var_size = 1.59595
data_size256 = [var_size*1*ngpus, var_size*2*ngpus, var_size*4*ngpus, var_size*6*ngpus, var_size*8*ngpus, var_size*10*ngpus]
io_time256 = [
      [5.11462, 6.82543, 9.96762, 13.1279, 16.5559, 27.4693], # no compression write
      [3.54099, 9.14661, 26.0711, 35.2579, 54.1108, 91.5174], # no decompression read
      [1.17601, 1.70578, ], # no prefetch compress
      [0.841577, 1.49374, ], # no prefetch decompress
      [], # prefetch compress
      [], # prefetch decompress
      [0.297585, 0.332137, ], # write
      [0.247754, 0.379161, ], # read
]

# Summit: 12 GPUs 12 writers 2 nodes
# Accumulation: 10
# Prefetch: on
# S: inf
eb = [1e13, 1e14, 1e15, 1e16, 1e17, 1e18]
io_time_eb = [
      [5.28733, 9.46398, 18.7111, 23.9214, 30.2847, 36.2648], #CR
      [1.77134, 1.69392, 1.67659, 1.6657, 1.74976, 1.65605], # prefetch compress
      [1.95462, 1.83119, 1.81801, 1.73798, 1.72375, 1.66943], # prefetch decompress
      [6.32398, 3.77037, 2.0365, 1.47153, 0.983315, 0.625277], # write
      [2.75943, 1.23801, 0.627058, 0.490939, 0.391299, 0.32388], # read

 ]
  
def plot_throughput_vs_datasize():
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
  x_idx = np.array(range(0, len(data_size1), 1))
  y_idx = np.array(range(0, 60, 5))
  throughput = np.asarray(data_size1)/np.asarray(io_time1[0])
  p1 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'dodgerblue', label='GPFS write')
  throughput = np.asarray(data_size1)/np.asarray(io_time1[1])
  p1 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'dodgerblue', label='GPFS read')
  throughput = np.asarray(data_size1)/np.asarray(io_time1[2])
  p1 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'maroon', label='comp. w/o opt')
  throughput = np.asarray(data_size1)/np.asarray(io_time1[3])
  p1 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'maroon', label='decomp. w/o opt')
  throughput = np.asarray(data_size1)/np.asarray(io_time1[4])
  p2 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'olivedrab', label='comp. w/ opt')
  throughput = np.asarray(data_size1)/np.asarray(io_time1[5])
  p2 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'olivedrab', label='decomp. w/ opt')
  x_ticks = []
  for i in range(len(data_size1)):
    x_ticks.append("{:.0f}".format(data_size1[i]) + "GB")
  ax.set_xticks(x_idx)
  ax.set_xticklabels(x_ticks)
  ax.set_xlabel("Data Size")
  ax.set_yticks(y_idx)
  ax.set_yscale('linear')
  ax.set_title('6 NVIDIA V100 GPUs')
  ax.set_ylabel('Throughput (GB/s)')
  ax.grid(which='major', axis='y')
  lgd = fig.legend(loc = 'center', bbox_to_anchor=(0.06, 0.45, 0.5, 0.5))
  # plt.tight_layout()
  plt.savefig('throughput_vs_datasize.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

plot_throughput_vs_datasize()


def plot_throughput_vs_num_gpus():
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
  x_idx = np.array(range(0, 4, 1))
  y_idx = np.array(range(0, 60, 10))
  size_idx = 5
  throughput = [
                np.asarray(data_size1_1[size_idx])/np.asarray(io_time1_1[2][size_idx]),
                np.asarray(data_size1_2[size_idx])/np.asarray(io_time1_2[2][size_idx]),
                np.asarray(data_size1_4[size_idx])/np.asarray(io_time1_4[2][size_idx]),
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[2][size_idx]),
              ]
  p1 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'maroon', label='comp. w/o opt')
  throughput = [
                np.asarray(data_size1_1[size_idx])/np.asarray(io_time1_1[3][size_idx]),
                np.asarray(data_size1_2[size_idx])/np.asarray(io_time1_2[3][size_idx]),
                np.asarray(data_size1_4[size_idx])/np.asarray(io_time1_4[3][size_idx]),
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[3][size_idx]),
              ]
  p1 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'maroon', label='decomp. w/o opt')
  throughput = [
                np.asarray(data_size1_1[size_idx])/np.asarray(io_time1_1[4][size_idx]),
                np.asarray(data_size1_2[size_idx])/np.asarray(io_time1_2[4][size_idx]),
                np.asarray(data_size1_4[size_idx])/np.asarray(io_time1_4[4][size_idx]),
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[4][size_idx]),
              ]
  p2 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'olivedrab', label='comp. w/ opt')
  throughput = [
                np.asarray(data_size1_1[size_idx])/np.asarray(io_time1_1[5][size_idx]),
                np.asarray(data_size1_2[size_idx])/np.asarray(io_time1_2[5][size_idx]),
                np.asarray(data_size1_4[size_idx])/np.asarray(io_time1_4[5][size_idx]),
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[5][size_idx]),
              ]
  p2 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'olivedrab', label='decomp. w/ opt')
  x_ticks = ['1', '2', '4', '6']
  ax.set_xticks(x_idx)
  ax.set_xticklabels(x_ticks)
  ax.set_xlabel("Number of GPUs")
  ax.set_yticks(y_idx)
  # ax.set_yscale('linear')
  ax.set_title("{:.0f} GB per GPU".format(data_size1[size_idx]))
  ax.set_ylabel('Throughput (GB/s)')
  ax.grid(which='major', axis='y')
  # lgd = fig.legend(loc = 'center', bbox_to_anchor=(0.1, 0.5, 0.5, 0.5))
  # plt.tight_layout()
  plt.savefig('throughput_vs_num_gpus.png', bbox_extra_artists=( ), bbox_inches='tight')

plot_throughput_vs_num_gpus()


def plot_throughput_vs_scale():
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
  x_idx = np.array(range(0, 8, 1))
  y_idx = np.array(range(0, 10000, 1000))
  size_idx = 5
  throughput = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
  p1 = ax.plot(x_idx, throughput, linestyle = '--', marker = '', color = 'red', label='GPFS peak')
  throughput = [
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[0][size_idx]),
                np.asarray(data_size2[size_idx])/np.asarray(io_time2[0][size_idx]),
                np.asarray(data_size4[size_idx])/np.asarray(io_time4[0][size_idx]),
                np.asarray(data_size8[size_idx])/np.asarray(io_time8[0][size_idx]),
                np.asarray(data_size16[size_idx])/np.asarray(io_time16[0][size_idx]),
                np.asarray(data_size32[size_idx])/np.asarray(io_time32[0][size_idx]),
                np.asarray(data_size64[size_idx])/np.asarray(io_time64[0][size_idx]),
                np.asarray(data_size128[size_idx])/np.asarray(io_time128[0][size_idx]),
              ]
  p1 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'dodgerblue', label='GPFS write')
  throughput = [
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[1][size_idx]),
                np.asarray(data_size2[size_idx])/np.asarray(io_time2[1][size_idx]),
                np.asarray(data_size4[size_idx])/np.asarray(io_time4[1][size_idx]),
                np.asarray(data_size8[size_idx])/np.asarray(io_time8[1][size_idx]),
                np.asarray(data_size16[size_idx])/np.asarray(io_time16[1][size_idx]),
                np.asarray(data_size32[size_idx])/np.asarray(io_time32[1][size_idx]),
                np.asarray(data_size64[size_idx])/np.asarray(io_time64[1][size_idx]),
                np.asarray(data_size128[size_idx])/np.asarray(io_time128[1][size_idx]),
              ]
  p1 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'dodgerblue', label='GPFS read')
  throughput = [
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[2][size_idx]),
                np.asarray(data_size2[size_idx])/np.asarray(io_time2[2][size_idx]),
                np.asarray(data_size4[size_idx])/np.asarray(io_time4[2][size_idx]),
                np.asarray(data_size8[size_idx])/np.asarray(io_time8[2][size_idx]),
                np.asarray(data_size16[size_idx])/np.asarray(io_time16[2][size_idx]),
                np.asarray(data_size32[size_idx])/np.asarray(io_time32[2][size_idx]),
                np.asarray(data_size64[size_idx])/np.asarray(io_time64[2][size_idx]),
                np.asarray(data_size128[size_idx])/np.asarray(io_time128[2][size_idx]),
              ]
  p1 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'maroon', label='comp. w/o opt')
  throughput = [
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[3][size_idx]),
                np.asarray(data_size2[size_idx])/np.asarray(io_time2[3][size_idx]),
                np.asarray(data_size4[size_idx])/np.asarray(io_time4[3][size_idx]),
                np.asarray(data_size8[size_idx])/np.asarray(io_time8[3][size_idx]),
                np.asarray(data_size16[size_idx])/np.asarray(io_time16[3][size_idx]),
                np.asarray(data_size32[size_idx])/np.asarray(io_time32[3][size_idx]),
                np.asarray(data_size64[size_idx])/np.asarray(io_time64[3][size_idx]),
                np.asarray(data_size128[size_idx])/np.asarray(io_time128[3][size_idx]),
              ]
  p1 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'maroon', label='decomp. w/o opt')
  throughput = [
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[4][size_idx]),
                np.asarray(data_size2[size_idx])/np.asarray(io_time2[4][size_idx]),
                np.asarray(data_size4[size_idx])/np.asarray(io_time4[4][size_idx]),
                np.asarray(data_size8[size_idx])/np.asarray(io_time8[4][size_idx]),
                np.asarray(data_size16[size_idx])/np.asarray(io_time16[4][size_idx]),
                np.asarray(data_size32[size_idx])/np.asarray(io_time32[4][size_idx]),
                np.asarray(data_size64[size_idx])/np.asarray(io_time64[4][size_idx]),
                np.asarray(data_size128[size_idx])/np.asarray(io_time128[4][size_idx]),
              ]
  p2 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'olivedrab', label='comp. w/ opt')
  throughput = [
                np.asarray(data_size1[size_idx])/np.asarray(io_time1[5][size_idx]),
                np.asarray(data_size2[size_idx])/np.asarray(io_time2[5][size_idx]),
                np.asarray(data_size4[size_idx])/np.asarray(io_time4[5][size_idx]),
                np.asarray(data_size8[size_idx])/np.asarray(io_time8[5][size_idx]),
                np.asarray(data_size16[size_idx])/np.asarray(io_time16[5][size_idx]),
                np.asarray(data_size32[size_idx])/np.asarray(io_time32[5][size_idx]),
                np.asarray(data_size64[size_idx])/np.asarray(io_time64[5][size_idx]),
                np.asarray(data_size128[size_idx])/np.asarray(io_time128[5][size_idx]),
              ]
  p2 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'olivedrab', label='decomp. w/ opt')
  x_ticks = ['6', '12', '24', '48', '96', '192', '384', '768']
  ax.set_xticks(x_idx)
  ax.set_xticklabels(x_ticks)
  ax.set_xlabel("Number of GPUs (ADIOS2 writers/readers)")
  # ax.set_yticks(y_idx)
  ax.set_yscale('log')
  ax.set_title("{:.0f} GB per GPU (MPI rank)".format(data_size1[size_idx]))
  ax.set_ylabel('Throughput (GB/s)')
  ax.grid(which='major', axis='y')
  lgd = fig.legend(loc = 'center', bbox_to_anchor=(0.11, 0.4, 0.5, 0.5))
  # plt.tight_layout()
  plt.savefig('throughput_vs_scale.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

plot_throughput_vs_scale()

def plot_io_time_vs_datasize():
  width = 0.15
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
  x_idx = np.array(range(0, len(data_size1), 1))
  y_idx = np.array(range(0, 60, 5))

  p1 = ax.bar(x_idx, io_time64[0], width, color = 'dodgerblue', label='write original')
  p2 = ax.bar(x_idx+width, io_time64[6], width, color = 'cyan', label = 'write compressed')
  p2 = ax.bar(x_idx+width, io_time64[2], width, color = 'maroon', bottom = io_time64[6], label='compression w/o opt')
  p2 = ax.bar(x_idx+width*2, io_time64[6], width, color = 'cyan')
  p2 = ax.bar(x_idx+width*2, io_time64[4], width, color = 'olivedrab', bottom = io_time64[6], label='compression w/ opt')
  
  x_ticks = []
  for i in range(len(data_size64)):
    x_ticks.append("{:.2f}".format(data_size64[i]/1000) + "TB")
  ax.set_xticks(x_idx+width)
  ax.set_xticklabels(x_ticks)
  ax.set_xlabel("Data Size (XGC data)")
  ax.set_yticks(y_idx)
  ax.set_yscale('linear')
  ax.set_title('384 NVIDIA V100 GPUs + 384 ADIOS2 Writers')
  ax.set_ylabel('Total I/O Time (s)')
  ax.grid(which='major', axis='y')
  lgd = fig.legend(loc = 'center', bbox_to_anchor=(0.13, 0.5, 0.5, 0.5))
  # plt.tight_layout()
  plt.savefig('io_time_vs_datasize.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

plot_io_time_vs_datasize()

def plot_io_time_vs_scale():
  width = 0.15
  size_idx = 5
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
  x_idx = np.array(range(0, 7, 1))
  y_idx = np.array(range(0, 60, 5))
  io_time_write = [io_time1[0][size_idx],
              io_time2[0][size_idx],
              io_time4[0][size_idx],
              io_time8[0][size_idx],
              io_time16[0][size_idx],
              io_time32[0][size_idx],
              io_time64[0][size_idx],
              ]
  p1 = ax.bar(x_idx, io_time_write, width, color = 'dodgerblue', label='write original')
  io_time_comp_write = [io_time1[6][size_idx],
                          io_time2[6][size_idx],
                          io_time4[6][size_idx],
                          io_time8[6][size_idx],
                          io_time16[6][size_idx],
                          io_time32[6][size_idx],
                          io_time64[6][size_idx],
                          ]
  p2 = ax.bar(x_idx+width, io_time_comp_write, width, color = 'cyan', label = 'write compressed')
  io_time_comp = [io_time1[2][size_idx],
              io_time2[2][size_idx],
              io_time4[2][size_idx],
              io_time8[2][size_idx],
              io_time16[2][size_idx],
              io_time32[2][size_idx],
              io_time64[2][size_idx],
              ]
  p2 = ax.bar(x_idx+width, io_time_comp, width, color = 'maroon', bottom = io_time_comp_write, label='compression w/o opt')
  
  io_time_comp = [io_time1[4][size_idx],
              io_time2[4][size_idx],
              io_time4[4][size_idx],
              io_time8[4][size_idx],
              io_time16[4][size_idx],
              io_time32[4][size_idx],
              io_time64[4][size_idx],
              ]
  p2 = ax.bar(x_idx+width*2, io_time_comp, width, color = 'olivedrab', bottom = io_time_comp_write, label='compression w/ opt')
  io_time_comp_write = [io_time1[6][size_idx],
                          io_time2[6][size_idx],
                          io_time4[6][size_idx],
                          io_time8[6][size_idx],
                          io_time16[6][size_idx],
                          io_time32[6][size_idx],
                          io_time64[6][size_idx],
                          ]
  p2 = ax.bar(x_idx+width*2, io_time_comp_write, width, color = 'cyan')
  x_ticks = ['6', '12', '24', '48', '96', '192', '384']
  ax.set_xticks(x_idx+width)
  ax.set_xticklabels(x_ticks)
  ax.set_xlabel("Number of GPUs (ADIOS2 writer/reads)")
  ax.set_yticks(y_idx)
  ax.set_yscale('linear')
  ax.set_title("{:.0f} GB per GPU (MPI rank)".format(data_size1[size_idx]))
  ax.set_ylabel('Total I/O Time (s)')
  ax.grid(which='major', axis='y')
  lgd = fig.legend(loc = 'center', bbox_to_anchor=(0.13, 0.5, 0.5, 0.5))
  # plt.tight_layout()
  plt.savefig('io_time_vs_scale.png', bbox_extra_artists=(lgd, ), bbox_inches='tight')

plot_io_time_vs_scale()


