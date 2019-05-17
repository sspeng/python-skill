import math
import numpy as np
import pandas as pd
from numpy import nan as NaN
import time
import datetime
import re

content = 'Citizen wang , always fall in love with neighbour，WANG'
rr = re.compile(r'wan\w', re.I) # 不区分大小写
print(rr)
print(type(rr))