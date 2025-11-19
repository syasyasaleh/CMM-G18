#-------------------- Experiment 1b without Teflon ---------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


# Time
t = np.linspace(0, 6, 1000)

# Read Excel file (replace with your filename)
df = pd.read_excel('berwick_bank.xlsx')

# Display the first few rows
print(df.head())

t=df["Time"].to_numpy(); x=df["Wind Speed"].to_numpy()
peaks,_=find_peaks(x,prominence=np.std(x)*0.2,distance=10)
peak_times=t[peaks]; peak_x=x[peaks]
if len(peak_times)>=2: T=np.mean(np.diff(peak_times)); f=1.0/T
else: T=0; f=0

#popt, _ = curve_fit(realistic_power, L, x, p0=[0.01, 0.01])
#P_fit = realistic_power(t, *popt)


plt.figure(figsize=(8, 4))
plt.plot(df['Time'], df['Wind Speed'], 
         marker='o',       # shape of marker (circle)
         linestyle='',      # '' means only markers, no line (optional)
         markersize=1,      # make the dots smaller
         label='Experimental Data')



# plt.plot(peak_times,peak_x,'ro',label='Peaks')
plt.text(0.02,0.98,f'T={T:.3f}s\nf={f:.3f}Hz',transform=plt.gca().transAxes,va='top',bbox=dict(boxstyle='round',facecolor='white'))
plt.tight_layout()
plt.title('Experimental Data - Condition a)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.grid(True)
plt.show()

