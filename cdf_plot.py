#
#error_naive_= error_naive
#error_sample_bias_ = error_sample_bias
#error_metric_ = error_metric
#error_metric_plus_sample_ = error_metric_plus_sample

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def plot_cdf_(data, num_bins, marker=None, title=None, xlabel=None):
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  counts, bin_edges = np.histogram (data, bins=num_bins)
  cdf = np.cumsum(counts)
  
  sns.set_style("whitegrid")
  if marker:
    ax, = plt.plot (bin_edges[1:], cdf/cdf[-1], marker, lw=2, markevery=5)
  else:
    ax, = plt.plot (bin_edges[1:], cdf/cdf[-1])
    
  plt.ylabel('CDF')
  
  if title:
      plt.title(title)
  if xlabel:
      plt.xlabel(xlabel)
  return ax

#plt.style.use("ggplot")
bg_color = 'white'
fg_color = 'black'
f = plt.figure(figsize=(6,5),facecolor=bg_color, edgecolor=fg_color)

p0 = plot_cdf_(error_normal, 100, marker = '*-')        
p1 = plot_cdf_(error_naive, 100, marker = 'o-')
p2 = plot_cdf_(error_sample_bias, 100, marker = 's-')    
p3 = plot_cdf_(error_metric, 100, marker = '^-')    
p4 = plot_cdf_(error_metric_plus_sample, 100, marker = 'P-') 



extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

p6 = plot_cdf_(error_naive_f, 100, marker = 'o:')
p7 = plot_cdf_(error_sample_bias_f, 100, marker = 's:')    
p8 = plot_cdf_(error_metric_f, 100, marker = '^:')    
p9 = plot_cdf_(error_metric_plus_sample_f, 100, marker = 'P:')        
l1 = plt.legend([p0, p1, p2, p3, p4],["No TL ","NL", "SSBC", "ML", "ML/SSBC"], loc = 1)
l2 = plt.legend([extra ,p6,p7,p8,p9],["post-FT:", "NL", "SSBC", "ML", "ML/SSBC"], loc = 4)    
plt.gca().add_artist(l1)
  
plt.xlabel("Localization error [m]")
plt.ylim([0,1])

f.savefig("cdf_plot_B0f2f3.pdf", bbox_inches='tight')

from matplotlib2tikz import save as tikz_save
tikz_save("cdf_plot.tex", standalone_environment= True)


