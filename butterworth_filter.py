# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# from scipy import signal    ##filter


# filename = 'gait1.csv'
# gait = pd.read_csv(filename)
# columns = gait.columns
# if 'ay (m/s^2)' in columns:
#     gait = gait.rename(columns={"ay (m/s^2)": "ay"})
# # print(gait)


# #time decision
# gait = gait.loc[(gait['time']>10)&(gait['time']<20)]

# #butter filter
# b, a = signal.butter(3, 0.05, btype='lowpass', analog=False)
# lowpass = signal.filtfilt(b, a, gait['ay'])

# plt.subplot(2,1,1)
# plt.plot(gait['time'],  gait['ay'], 'b-')
# plt.xlabel("Time(sec)")
# plt.ylabel("Acce on y-axis(m/s^2)")
# plt.title(" Signal before filtering")
# plt.subplot(2,1,2)
# plt.plot(gait['time'], lowpass, 'r-')
# plt.xlabel("Time(sec)")
# plt.ylabel("Acce on y-axis(m/s^2)")
# plt.title(" Signal after filtering")

# plt.tight_layout()
# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal    ##filter


filename = 'data/sensor.csv'
gait = pd.read_csv(filename)
columns = gait.columns
if 'ay (m/s^2)' in columns:
    gait = gait.rename(columns={"ay (m/s^2)": "ay"})
print(gait)


gait = gait.loc[(gait['time']>12)&(gait['time']<44)]

#butter filter
b, a = signal.butter(3, 0.08, btype='lowpass', analog=False)
lowpass = signal.filtfilt(b, a, gait['ay'])

plt.subplot(2,1,1)
plt.plot(gait['time'],  gait['ay'], 'b-')
plt.xlabel("Time(sec)")
plt.ylabel("Acce on y-axis(m/s^2)")
plt.title(" Signal before filtering")
plt.subplot(2,1,2)
plt.plot(gait['time'], lowpass, 'r-')
plt.xlabel("Time(sec)")
plt.ylabel("Acce on y-axis(m/s^2)")
plt.title(" Signal after filtering")

plt.tight_layout()
plt.show()