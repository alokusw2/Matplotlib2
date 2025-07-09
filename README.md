# Matplotlib2

1 Problem 1 : Matplotlib	Line Chart in Matplotlib	(https://www.geeksforgeeks.org/line-chart-in-matplotlib-python/	)	


------------------------------------------------

#Simple Line Plot
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4])  # X-axis 
y = x*2  # Y-axis

plt.plot(x, y)  
plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4])
y = x*2

plt.plot(x, y)
plt.xlabel("X-axis")        # Label for the X-axis
plt.ylabel("Y-axis")        # Label for the Y-axis
plt.title("Any suitable title")  # Chart title
plt.show()

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', linestyle='-')

# Add annotations
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(0, 10), ha='center')

plt.title('Line Chart with Annotations')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import numpy as np


x = np.array([1, 2, 3, 4])
y = x*2

plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Any suitable title")
plt.show()  # show first chart

plt.figure()
x1 = [2, 4, 6, 8]
y1 = [3, 5, 7, 9]
plt.plot(x1, y1, '-.')
plt.show()


import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4])
y = x*2

# first plot with X and Y data
plt.plot(x, y)

x1 = [2, 4, 6, 8]
y1 = [3, 5, 7, 9]

# second plot with x1 and y1 data
plt.plot(x1, y1, '-.')

plt.xlabel("X-axis data")
plt.ylabel("Y-axis data")
plt.title('multiple plots')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 2, 3, 4])
y = x*2

plt.plot(x, y)

x1 = [2, 4, 6, 8]
y1 = [3, 5, 7, 9]

plt.plot(x, y1, '-.')
plt.xlabel("X-axis data")
plt.ylabel("Y-axis data")
plt.title('multiple plots')

plt.fill_between(x, y, y1, color='green', alpha=0.5)
plt.show()




------------------------------------------------


2 Problem 2 : Matplotlib	Histogram in Matplotlib	(https://www.geeksforgeeks.org/plotting-histogram-in-python-using-matplotlib/	)


------------------------------------------------
#Histogram in Matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Generate random data for the histogram
data = np.random.randn(1000)

# Plotting a basic histogram
plt.hist(data, bins=30, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')

# Display the plot
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate random data for the histogram
data = np.random.randn(1000)

# Creating a customized histogram with a density plot
sns.histplot(data, bins=30, kde=True, color='lightgreen', edgecolor='red')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Customized Histogram with Density Plot')

# Display the plot
plt.show()



import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# Creating dataset
np.random.seed(23685752)
N_points = 10000
n_bins = 20

# Creating distribution
x = np.random.randn(N_points)
y = .8 ** x + np.random.randn(10000) + 25
legend = ['distribution']

# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7), 
                        tight_layout = True)


# Remove axes splines 
for s in ['top', 'bottom', 'left', 'right']: 
    axs.spines[s].set_visible(False) 

# Remove x, y ticks
axs.xaxis.set_ticks_position('none') 
axs.yaxis.set_ticks_position('none') 
  
# Add padding between axes and labels 
axs.xaxis.set_tick_params(pad = 5) 
axs.yaxis.set_tick_params(pad = 10) 

# Add x, y gridlines 
axs.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.5, 
        alpha = 0.6) 

# Add Text watermark 
fig.text(0.9, 0.15, 'Jeeteshgavande30', 
         fontsize = 12, 
         color ='red',
         ha ='right',
         va ='bottom', 
         alpha = 0.7) 

# Creating histogram
N, bins, patches = axs.hist(x, bins = n_bins)

# Setting color
fracs = ((N**(1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# Adding extra features    
plt.xlabel("X-axis")
plt.ylabel("y-axis")
plt.legend(legend)
plt.title('Customized histogram')

# Show plot
plt.show()



#Multiple Histograms with Subplots

import matplotlib.pyplot as plt
import numpy as np

# Generate random data for multiple histograms
data1 = np.random.randn(1000)
data2 = np.random.normal(loc=3, scale=1, size=1000)

# Creating subplots with multiple histograms
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

axes[0].hist(data1, bins=30, color='Yellow', edgecolor='black')
axes[0].set_title('Histogram 1')

axes[1].hist(data2, bins=30, color='Pink', edgecolor='black')
axes[1].set_title('Histogram 2')

# Adding labels and title
for ax in axes:
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')

# Adjusting layout for better spacing
plt.tight_layout()

# Display the figure
plt.show()

#Stacked Histogram using Matplotlib

import matplotlib.pyplot as plt
import numpy as np

# Generate random data for stacked histograms
data1 = np.random.randn(1000)
data2 = np.random.normal(loc=3, scale=1, size=1000)

# Creating a stacked histogram
plt.hist([data1, data2], bins=30, stacked=True, color=['cyan', 'Purple'], edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Stacked Histogram')

# Adding legend
plt.legend(['Dataset 1', 'Dataset 2'])

# Display the plot
plt.show()


#Plot 2D Histogram (Hexbin Plot) using Matplotlib

import matplotlib.pyplot as plt
import numpy as np

# Generate random 2D data for hexbin plot
x = np.random.randn(1000)
y = 2 * x + np.random.normal(size=1000)

# Creating a 2D histogram (hexbin plot)
plt.hexbin(x, y, gridsize=30, cmap='Blues')

# Adding labels and title
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('2D Histogram (Hexbin Plot)')

# Adding colorbar
plt.colorbar()

# Display the plot
plt.show()





------------------------------------------------


3 Problem 3 : Matplotlib	Scatterplot in Matplotlib	(	https://www.geeksforgeeks.org/matplotlib-pyplot-scatter-in-python/)	


------------------------------------------------
#Matplotlib Scatter

import matplotlib.pyplot as plt
import numpy as np

x = np.array([12, 45, 7, 32, 89, 54, 23, 67, 14, 91])
y = np.array([99, 31, 72, 56, 19, 88, 43, 61, 35, 77])

plt.scatter(x, y)
plt.title("Basic Scatter Plot")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()

#Example 1

x1 = np.array([160, 165, 170, 175, 180, 185, 190, 195, 200, 205])
y1 = np.array([55, 58, 60, 62, 64, 66, 68, 70, 72, 74])

x2 = np.array([150, 155, 160, 165, 170, 175, 180, 195, 200, 205])
y2 = np.array([50, 52, 54, 56, 58, 64, 66, 68, 70, 72])

plt.scatter(x1, y1, color='blue', label='Group 1')
plt.scatter(x2, y2, color='red', label='Group 2')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Comparison of Height vs Weight between two groups')
plt.legend()
plt.show()

##Example 2

x = np.array([3, 12, 9, 20, 5, 18, 22, 11, 27, 16])
y = np.array([95, 55, 63, 77, 89, 50, 41, 70, 58, 83])

a = [20, 50, 100, 200, 500, 1000, 60, 90, 150, 300] # size
b = ['red', 'green', 'blue', 'purple', 'orange', 'black', 'pink', 'brown', 'yellow', 'cyan'] # color

plt.scatter(x, y, s=a, c=b, alpha=0.6, edgecolors='w', linewidth=1)
plt.title("Scatter Plot with Varying Colors and Sizes")
plt.show()

#Example 3


x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
sizes = [30, 80, 150, 200, 300]  # Bubble sizes

plt.scatter(x, y, s=sizes, alpha=0.5, edgecolors='blue', linewidths=2)
plt.title("Bubble Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


#Example 4:


x = np.random.randint(50, 150, 100)
y = np.random.randint(50, 150, 100)
colors = np.random.rand(100)  # Random float values for color mapping
sizes = 20 * np.random.randint(10, 100, 100)

plt.scatter(x, y, c=colors, s=sizes, cmap='viridis', alpha=0.7)
plt.colorbar(label='Color scale')
plt.title("Scatter Plot with Colormap and Colorbar")
plt.show()

#Example 5

plt.scatter(x, y, marker='^', color='magenta', s=100, alpha=0.7)
plt.title("Scatter Plot with Triangle Markers")
plt.show()





------------------------------------------------


4 Problem 4 : Matplotlib	Pie Chart in Matplotlib	(	https://www.geeksforgeeks.org/plot-a-pie-chart-in-python-using-matplotlib/		)												
------------------------------------------------

#Pie Chart in Python using Matplotlib

# Import libraries
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
cars = ['AUDI', 'BMW', 'FORD',
        'TESLA', 'JAGUAR', 'MERCEDES']

data = [23, 17, 35, 29, 12, 41]

# Creating plot
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=cars)

# show plot
plt.show()


#Customizing Pie Charts

# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Creating dataset
cars = ['AUDI', 'BMW', 'FORD',
        'TESLA', 'JAGUAR', 'MERCEDES']

data = [23, 17, 35, 29, 12, 41]


# Creating explode data
explode = (0.1, 0.0, 0.2, 0.3, 0.0, 0.0)

# Creating color parameters
colors = ("orange", "cyan", "brown",
          "grey", "indigo", "beige")

# Wedge properties
wp = {'linewidth': 1, 'edgecolor': "green"}

# Creating autocpt arguments


def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)


# Creating plot
fig, ax = plt.subplots(figsize=(10, 7))
wedges, texts, autotexts = ax.pie(data,
                                  autopct=lambda pct: func(pct, data),
                                  explode=explode,
                                  labels=cars,
                                  shadow=True,
                                  colors=colors,
                                  startangle=90,
                                  wedgeprops=wp,
                                  textprops=dict(color="magenta"))

# Adding legend
ax.legend(wedges, cars,
          title="Cars",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")
ax.set_title("Customizing pie chart")

# show plot
plt.show()


#Creating a Nested Pie Chart in Python

# Import libraries
from matplotlib import pyplot as plt
import numpy as np


# Creating dataset
size = 6
cars = ['AUDI', 'BMW', 'FORD',
        'TESLA', 'JAGUAR', 'MERCEDES']

data = np.array([[23, 16], [17, 23],
                 [35, 11], [29, 33],
                 [12, 27], [41, 42]])

# normalizing data to 2 pi
norm = data / np.sum(data)*2 * np.pi

# obtaining ordinates of bar edges
left = np.cumsum(np.append(0,
                           norm.flatten()[:-1])).reshape(data.shape)

# Creating color scale
cmap = plt.get_cmap("tab20c")
outer_colors = cmap(np.arange(6)*4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9,
                              10, 12, 13, 15,
                              17, 18, 20]))

# Creating plot
fig, ax = plt.subplots(figsize=(10, 7),
                       subplot_kw=dict(polar=True))

ax.bar(x=left[:, 0],
       width=norm.sum(axis=1),
       bottom=1-size,
       height=size,
       color=outer_colors,
       edgecolor='w',
       linewidth=1,
       align="edge")

ax.bar(x=left.flatten(),
       width=norm.flatten(),
       bottom=1-2 * size,
       height=size,
       color=inner_colors,
       edgecolor='w',
       linewidth=1,
       align="edge")

ax.set(title="Nested pie chart")
ax.set_axis_off()

# show plot
plt.show()



------------------------------------------------

