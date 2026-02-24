# Hadronic-Tau-Detector-ML
A supervised learning model to distinguish hadronic tau decays from QCD jets using high-energy physics event data using the Tensorflow framework.

Hadronic tau decays produce narrow, low-multiplicity jets that can be easily confused with QCD background jets. Efficient tau identification is essential in many LHC analyses, including Higgs boson decay channels.

# How the Model Works
The model works by reading the root file of the CERN Monte Carlo simulation data

The model takes a few key variables:

     - tau_pt = transverse momentum of the tau
     
     - tau_eta = pseudorapidity (detector location in comparison)
     
     - tau_nTracks = number of tracks associated with the tau
     
     - tau_phi = azimuthal angle
     
     - tau_charge = charge of the tau
     
     - tau_E = energy of the tau

We use these variables to train the model to make a prediction against the MC_truth_label (this is the true identity of the tau candidate)

The model generates an ROC curve, plotting the True positive rate against the False positive rate and generating an AUC score to give us an idea to its accuracy on raw detector data. It currently sits at a 0.710


# Extra Information
Please note you require the https://opendata.cern.ch/record/15002 from CERN open data

Make sure you change the placeholder 'filepath' to the filepaths of your root files
