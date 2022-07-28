import scipy.io as sio
from scipy import signal
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np
import pywt
import time

# ---Define function to compute instantaneous median frequency---
def Compute_Median_Frequency(norm2, Wave_FreqS):
    MedianFreq = np.empty([norm2.shape[1], 1])
    for iTime in range(norm2.shape[1]):
        totinteg = np.trapz(norm2[:, iTime])
        Fmedsup = np.array(np.where(integrate.cumtrapz(norm2[:, iTime]) > 0.5 * totinteg))
        MedianFreq[iTime] = Wave_FreqS[Fmedsup[0, 0]]
    return MedianFreq

Participants = ['P' + str(iP) for iP in range(1, 25)] # Define Participants using comprehension list
Muscles = ['Dent1', 'TrapInf', 'Bi', 'Tri', 'Dent2', 'DeltA', 'DeltM',
               'DeltP', 'TrapSup', 'TrapMed']
Freq = 1000

for iP in range(len(Participants)):
    # EMG_filtered = sio.loadmat("J:/IRSST_Fatigue/Pointage_repetitif/EMG_Pointage_Python/" + Participants[iP] + ".mat") # To import the mat structure saved using python
    MatStruc = sio.loadmat("J:/IRSST_Fatigue/Pointage_repetitif_EMG/EMG_Clean/" + Participants[iP] + "_Pointage.mat", squeeze_me=True, struct_as_record=False) # To import the mat structure saved using matlab
    EMG_filtered = MatStruc['EMG'].data
    Muscles = MatStruc['EMG'].Muscles
    TFR = {}
    tfr = {}
    MedianFreq = {}
    # # ---For mat structure saved using python---
    # k = 1000
    # while k > 1:
    #     print(k)
    #     Length_signal = int(EMG_filtered['Bi'].shape[1]/k)
    #     t, dt = np.linspace(0, Length_signal/Freq, Length_signal, retstep=True)
    #     sig = EMG_filtered['Bi'][0, 0:Length_signal]
    #     plt.figure(1)
    #     plt.plot(t, sig)
    #     plt.show()
    #     fs = 1/dt
    #     w = 7.
    #     freq = np.linspace(1, fs/5, Length_signal)
    #     widths = w*fs / (2*freq*np.pi)

    for iM in range(len(Muscles)):
        # ---For mat structure saved using matlab---
        k = 1000
        while k >= 1000:
            print(k)
            Length_signal = int(EMG_filtered[:, iM].shape[0] / k)
            t, dt = np.linspace(0, Length_signal / Freq, Length_signal, retstep=True)
            sig = EMG_filtered[0:Length_signal, iM]
            # plt.figure(1)
            # plt.plot(t, sig)
            # plt.show()
            fs = 1 / dt
            w = 7.
            freq = np.linspace(1, 400, Length_signal)
            widths = w * fs / (2 * freq * np.pi)

            start = time.time()
            cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
            norm = abs(cwtm)**2
            tfr.update({Muscles[iM]: norm})
            # plt.plot(freq, norm[:, -1]) # plot le spectre à l'instant 1
            plt.pcolormesh(t, freq, np.abs(cwtm)**2, cmap='viridis', shading='gouraud')
            end = time.time()
            print('Time to run TFR', end-start)
            # plt.show()
            k = k/2

            MedFreq = Compute_Median_Frequency(norm, freq)
            MedianFreq.update({Muscles[iM]: MedFreq})
            # plt.plot(MedFreq)
            # plt.show()

    TFR.update({'TFR': tfr, 'MedianFreq': MedianFreq})
    sio.savemat("J:/IRSST_Fatigue/Pointage_repetitif/EMG_Pointage_Python/TFR_" + Participants[iP] + ".mat", TFR)
    print('saved')
