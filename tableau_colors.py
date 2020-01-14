# Tableau 20 Colors
from collections import OrderedDict
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
             
# Tableau Color Blind 10
tableau10blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
             (255, 188, 121), (207, 207, 207)]
  
# Rescale to values between 0 and 1 and add hex
tableau20_hex = []
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)

    tableau20_hex.append('#%02x%02x%02x' % (r, g, b))
for i in range(len(tableau10blind)):  
    r, g, b = tableau10blind[i]  
    tableau10blind[i] = (r / 255., g / 255., b / 255.)
# Use with plt.plot(…, color=tableau[0],…)

tableau20_name = ['blue','light_blue','orange','light_orange','green',
	'light_green','red','light_red','purple','light_purple','brown','light_brown',
	'pink','light_pink','gray','light_gray','yellow','light_yellow','cyan','light_cyan']
	
tableau20 = OrderedDict(zip(tableau20_name, tableau20))
tableau20_hex = OrderedDict(zip(tableau20_name, tableau20_hex))
t20=tableau20
t20_hex=tableau20_hex


	

# https://stackoverflow.com/questions/32236046/add-a-legend-to-my-heatmap-plot
# http://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html
# From http://tableaufriction.blogspot.ro/2012/11/finally-you-can-use-tableau-data-colors.html
# With code from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
