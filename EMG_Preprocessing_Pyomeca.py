import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from pyomeca import Analogs
# import time

# Define variables
Participants = ['P' + str(iP) for iP in range(1, 2)] # Define Participants using comprehension list
Muscles = ['Dent1', 'TrapInf', 'Bi', 'Tri', 'Dent2', 'DeltA', 'DeltM',
               'DeltP', 'TrapSup', 'TrapMed']
EMG_Freq = 2000 # Acquisition frequency 2000 Hz

# Start the for loop
for iP in range(len(Participants)):
    # ---Import File---
    # Import .mat file
    StrucData = sio.loadmat("J:/IRSST_Fatigue/Pointage_repetitif/EMG_Pointage_Python/" + Participants[iP] + ".mat")
    Data_mat = np.array([StrucData[Muscles[iM]] for iM in range(len(Muscles))])
    EMG = Analogs(Data_mat[:, 0, :])
    EMG['channel'] = Muscles

    # # Import .c3d raw file (this is longer than opening a .mat file)
    # data_path = "J:/IRSST_Fatigue/Pointage_repetitif/Data/" + Participants[iP] + "/Trial/Pointage.c3d"
    # EMG = Analogs.from_c3d(data_path, suffix_delimiter=".", usecols=Muscles)

    # ---Raw signals---
    fft_EMG = EMG.meca.fft(freq=EMG_Freq)

    # # EMG.plot(x="time", col="channel", col_wrap=5)
    # # plt.suptitle('Raw EMG signals', fontsize=16)
    #
    # plt.figure(2)
    # plt.suptitle('FFT Raw EMG signals', fontsize=16)
    # for iM in range(len(Muscles)):
    #     plt.subplot(2, 5, iM + 1)
    #     fft_EMG[iM].plot.line(x="freq")
    #     plt.title(Muscles[iM])

    # ---Band-pass filter signals---
    low_cut = 10
    high_cut = 400
    EMGBP = EMG.meca.band_pass(order=2, cutoff=[low_cut, high_cut], freq=EMG_Freq)

    # fft_EMGBP = EMGBP.meca.fft(EMG_Freq)
    #
    # EMGBP.plot(x="time", col="channel", col_wrap=5)
    # plt.suptitle('band-pass filtered EMG signals', fontsize=16)
    #
    # plt.figure(4)
    # plt.suptitle('FFT band-pass filtered EMG signals', fontsize=16)
    # for iM in range(len(Muscles)):
    #     plt.subplot(2, 5, iM + 1)
    #     fft_EMGBP[iM].plot.line(x="freq")
    #     plt.title(Muscles[iM])

    # ---stop-band filter signals---
    low_cut = 59
    high_cut = 61
    EMGBS = EMGBP.meca.band_stop(order=2, cutoff=[low_cut, high_cut], freq=EMG_Freq)

    # fft_EMGBS = EMGBS.meca.fft(EMG_Freq)
    #
    # EMGBS.plot(x="time", col="channel", col_wrap=5)
    # plt.suptitle('stop-band filtered EMG signals', fontsize=16)
    #
    # plt.figure(6)
    # plt.suptitle('FFT stop-band filtered EMG signals', fontsize=16)
    # for iM in range(len(Muscles)):
    #     plt.subplot(2, 5, iM + 1)
    #     fft_EMGBS[iM].plot.line(x="freq")
    #     plt.title(Muscles[iM])

    # ---Remove mean---
    EMGBL = EMGBS
    EMGBL.meca.center()

    fft_EMGBL = EMGBL.meca.fft(EMG_Freq)

    EMGBL.plot(x="time", col="channel", col_wrap=5)
    plt.suptitle('Detrended & filtered EMG signals', fontsize=16)

    plt.figure(8)
    plt.suptitle('FFT detrended & filtered EMG signals', fontsize=16)
    for iM in range(len(Muscles)):
        plt.subplot(2, 5, iM + 1)
        fft_EMGBL[iM].plot.line(x="freq")
        plt.title(Muscles[iM])

    plt.show()
