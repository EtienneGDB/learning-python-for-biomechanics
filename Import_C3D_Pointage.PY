# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:55:01 2022

@author: p1098713
"""
# Import libraries
import ezc3d
import numpy as np
import pickle
import matplotlib.pyplot as plt
# import time

# Define variables
Participants = ['P' + str(iP) for iP in range(1, 17)] # Define Participants using comprehension list
Muscles = ['Dent1', 'TrapInf', 'Bi', 'Tri', 'Dent2', 'Dent3', 'Dent4', 'DeltA', 'DeltM',
           'DeltP', 'TrapSup', 'TrapMed']

for iP in range(18, 25):
    # Import .c3d raw file
    c3d = ezc3d.c3d('J:/IRSST_Fatigue/Pointage_repetitif/Data/' + Participants[iP] + '/Trial/Pointage.c3d')

    # Print the parameters
    # print("# ---- PARAMETERS ---- #")
    # print(f"Name of the analogs = {c3d['parameters']['ANALOG']['LABELS']['value']}")
    NumParam = c3d['parameters']['ANALOG']['USED']['value'][0] # Number of parameters recorded
    NameParam = c3d['parameters']['ANALOG']['LABELS']['value'] # Name of parameters recorded

    # Store what we want into data variable
    data = c3d['data']['analogs']

    # Plot all the data
    # plt.figure()
    # for i in range(NumParam):
    #     plt.subplot(5, 5, i+1)
    #     plt.plot(data[0, i, :])
    # plt.show()

    # Get the index of wanted parameters -> Muscles
    # start = time.time()
    # res = []
    # for iM in range(len(Muscles)):
    #     for iNameParam in range(NumParam):
    #         if NameParam[iNameParam].find(Muscles[iM]) == 0 and NameParam[iNameParam].find('3') == -1 and NameParam[iNameParam].find('4') == -1:
    #             id_Muscles.append(iNameParam)
    #
    # end = time.time()
    # print(end-start)
    # print(xxx)

    # List comprehension equivalent to the precedent loop
    id_Muscles = [iNameParam for iNameParam in range(NumParam) for iM in range(len(Muscles))
           if NameParam[iNameParam].find(Muscles[iM])==0 and NameParam[iNameParam].find('3') == -1 and NameParam[iNameParam].find('4') == -1]

    Name_id_Muscles = [Muscles[iM] for iNameParam in range(NumParam) for iM in range(len(Muscles))
    if NameParam[iNameParam].find(Muscles[iM]) == 0 and NameParam[iNameParam].find('3') == -1 and NameParam[iNameParam].find('4') == -1]

    data_to_keep = np.array(data[0, id_Muscles, :])
    StrucData = {x:list(y) for x,y in zip(Name_id_Muscles, zip(*data_to_keep.transpose()))}

    # Plot the wanted data
    if False:
        plt.figure(figsize=(9, 5))
        for iM in range(len(Name_id_Muscles)):
            plt.subplot(5, 2, iM+1)
            plt.plot(StrucData[Name_id_Muscles[iM]])
            plt.title(Name_id_Muscles[iM])
            plt.suptitle('Raw EMG signals', fontsize=16)

        plt.show()
        input("Press Enter to continue...")

    # Store the wanted data into pickle files
    f = open("J:/IRSST_Fatigue/Pointage_repetitif/EMG_Pointage_Python/" + Participants[iP] + ".p", "wb")
    pickle.dump(StrucData, f)
    f.close()
    