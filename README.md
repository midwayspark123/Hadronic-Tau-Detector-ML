# Hadronic-Tau-Detector-ML
A Neural Network trained on MC truth labels. Predicts whether hadronic tau candidates are real or just QCD jets.

      This is currently a work in progress:                                                                  
      |
      |_ Version 1 - Contains just the training of the neural network:
      |      |                                                                                                                                                                                                
      |      |_ ~98.8% accurate with classifying taus and non-taus                                                
      |      |  
      |      |_ The loss is high, feature normalisation needed
      |
      |_ Version 2 - Contains the training of the model and works on detector data:
             |
             |_ Fix adds ROC curve, removes tau_BDTid, adds detector data functionality, smaller loss
             |
             |_ ROC = 0.710 

Also, please note you require the https://opendata.cern.ch/record/15002 from CERN open data, please make sure the filepaths line up.

