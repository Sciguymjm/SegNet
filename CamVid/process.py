import os
from scipy.misc import imread
import numpy as np
for f in os.listdir("trainannot"):
	print imread("trainannot\\"+f)