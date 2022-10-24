import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv

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

def read_csv(filename):
  f = open(filename)
  csv_reader = csv.reader(f, delimiter=',')
  nrows = 0
  ncols = 0
  data = []
  for row in csv_reader:
    nrows += 1;
    ncols = len(row)
  data = np.zeros((nrows, ncols-1))
  f.seek(0)
  row_idx = 0;
  for row in csv_reader:
    col_idx = 0
    for item in row:
      # print(item)
      if (col_idx < ncols-1):
        data[row_idx, col_idx] = item
      col_idx += 1
    row_idx += 1
  return data

summit_data_1 = read_csv("./results/summit_xgc_1e14_1e17_inf_1.csv")
summit_data_2 = read_csv("./results/summit_xgc_1e14_1e17_inf_2.csv")
summit_data_4 = read_csv("./results/summit_xgc_1e14_1e17_inf_4.csv")
summit_data_6 = read_csv("./results/summit_xgc_1e14_1e17_inf_6.csv")
summit_data_12 = read_csv("./results/summit_xgc_1e14_1e17_inf_12.csv")
summit_data_24 = read_csv("./results/summit_xgc_1e14_1e17_inf_24.csv")
summit_data_48 = read_csv("./results/summit_xgc_1e14_1e17_inf_48.csv")
summit_data_96 = read_csv("./results/summit_xgc_1e14_1e17_inf_96.csv")
summit_data_192 = read_csv("./results/summit_xgc_1e14_1e17_inf_192.csv")
summit_data_384 = read_csv("./results/summit_xgc_1e14_1e17_inf_384.csv")

summit_data_single_node = np.stack([summit_data_1, summit_data_2, summit_data_4, summit_data_6])
summit_data_at_scale = np.stack([summit_data_6, summit_data_12, summit_data_24, summit_data_48,
                                 summit_data_96, summit_data_192, summit_data_384])

crusher_data_1 = read_csv("./results/crusher_xgc_1e14_1e17_inf_1.csv")
crusher_data_2 = read_csv("./results/crusher_xgc_1e14_1e17_inf_2.csv")
crusher_data_4 = read_csv("./results/crusher_xgc_1e14_1e17_inf_4.csv")
crusher_data_8 = read_csv("./results/crusher_xgc_1e14_1e17_inf_8.csv")
crusher_data_16 = read_csv("./results/crusher_xgc_1e14_1e17_inf_16.csv")
crusher_data_32 = read_csv("./results/crusher_xgc_1e14_1e17_inf_32.csv")
crusher_data_64 = read_csv("./results/crusher_xgc_1e14_1e17_inf_64.csv")
crusher_data_128 = read_csv("./results/crusher_xgc_1e14_1e17_inf_128.csv")
crusher_data_256 = read_csv("./results/crusher_xgc_1e14_1e17_inf_256.csv")

crusher_data_single_node = np.stack([crusher_data_1, crusher_data_2, crusher_data_4])
crusher_data_at_scale = np.stack([crusher_data_4, crusher_data_8, crusher_data_16, crusher_data_32,
                                  crusher_data_64, crusher_data_128, crusher_data_256])

# print crusher_data_single_node

var_size_summit = 1.59595 * 10;
var_size_crusher = 1.59595 * 6 * 2;

summit_num_gpus_single_node = np.array([1, 2, 4, 6])
summit_num_gpus_at_scale = np.array([6, 12, 24, 48, 96, 192, 384])#,  96, 192, 384, 768, 1536])

crusher_num_gpus_single_node = np.array([1, 2, 4])
crusher_num_gpus_at_scale = np.array([4, 8, 16, 32, 64, 128, 256])#, 32, 64, 128, 256, 512, 1024])


def plot_throughput(var_size, num_gpus, time_results, io_throughput, fs_name, machine, gpus_per_node, gpu_name, output):
  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))
  x_idx = np.arange(0, len(num_gpus), 1)
  # y_idx = np.arange(0, 60, 10)
  eb_idx = 3
  throughput = var_size * num_gpus / time_results[:, 2, eb_idx] / 1000.0
  p1 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'maroon', label='comp.')
  throughput = var_size * num_gpus / time_results[:, 3, eb_idx] / 1000.0
  p1 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'maroon', label='decomp.')
  throughput = var_size * num_gpus / time_results[:, 4, eb_idx] / 1000.0
  p2 = ax.plot(x_idx, throughput, linestyle = '-', marker = '.', color = 'olivedrab', label='comp. opt')
  throughput = var_size * num_gpus / time_results[:, 5, eb_idx] / 1000.0
  p2 = ax.plot(x_idx, throughput, linestyle = '--', marker = '.', color = 'olivedrab', label='decomp. opt')
  if (io_throughput != 0):
    throughput = np.full(len(num_gpus), io_throughput) / 1000.0
    p1 = ax.plot(x_idx, throughput, linestyle = '-', marker = '', color = 'red', linewidth = 4.0, label='I/O peak ({})'.format(fs_name))

  x_ticks = []
  for ngpus in num_gpus:
    x_ticks.append(str(ngpus/gpus_per_node))
  ax.set_xticks(x_idx)
  ax.set_xticklabels(x_ticks)
  ax.set_xlabel("Number of Nodes ({}*{}/Node)".format(gpus_per_node, gpu_name))
  # ax.set_yticks(y_idx)
  ax.set_yscale('linear')
  ax.set_title("{} (XGC {:.0f} GB per GPU)".format(machine, var_size))
  ax.set_ylabel('Throughput (TB/s)')
  ax.grid(which='major', axis='y')
  lgd = fig.legend(loc = 'center', bbox_to_anchor=(0.1, 0.43, 0.5, 0.5))
  plt.tight_layout()
  plt.savefig('{}.png'.format(output), bbox_extra_artists=( ), bbox_inches='tight')


# Frontier specs: https://www.ornl.gov/news/frontier-supercomputer-debuts-worlds-fastest-breaking-exascale-barrier

plot_throughput(var_size_crusher, crusher_num_gpus_single_node, 
                crusher_data_single_node, 0, "Orion", "Frontier/Crusher", 4, "AMD MI250X", "crusher_throughput_single_node")
plot_throughput(var_size_crusher, crusher_num_gpus_at_scale, 
                crusher_data_at_scale, 5000.0, "Orion", "Frontier/Crusher", 4, "AMD MI250X", "crusher_throughput_at_scale")
plot_throughput(var_size_summit, summit_num_gpus_single_node, 
                summit_data_single_node, 0, "Alpine", "Summit", 6, "NVIDIA V100", "summit_throughput_single_node")
plot_throughput(var_size_summit, summit_num_gpus_at_scale, 
                summit_data_at_scale, 2500.0, "Alpine", "Summit", 6, "NVIDIA V100", "summit_throughput_at_scale")

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

# plot_throughput_vs_scale()

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

# plot_io_time_vs_datasize()

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

# plot_io_time_vs_scale()


