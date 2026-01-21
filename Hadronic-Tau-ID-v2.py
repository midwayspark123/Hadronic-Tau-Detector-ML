import uproot
import numpy as np
import awkward as ak

#####################################################
####### TRAINING THE TAU IDENTIFICATION MODEL ####### 
##################################################### 

#Loading the root file
filepath = '/home/midway/CERN Open Data/1lep1tau/MC/mc_341123.ggH125_tautaulh.1lep1tau.root'
file = uproot.open(filepath)

print(file.keys())
tree = file['mini']

#Choosing the variables to use in the comparison
features = [
    'tau_pt', #tau_pt = energy scale
    'tau_eta', #tau_eta = pseudorapidity (detector location in comparison)
    'tau_nTracks', #tau_nTracks = number of tracks associated with the tau
    'tau_phi', #tau_phi = azimuthal angle
    'tau_charge', #tau_mcharge = charge of the tau
    'tau_E' #tau_E = energy of the tau

]

#Load tau branches into awkward arrays
arrays = tree.arrays(features + ["tau_truthMatched"], library='ak')

#create lists to store per-tau candidate
X_list = []
y_list = []

for pt_list, eta_list, nTracks_list, phi_list, charge_list, E_list, truth_list in zip(
    arrays['tau_pt'], arrays['tau_eta'], arrays['tau_nTracks'], arrays['tau_phi'], arrays['tau_charge'], arrays['tau_E'], arrays['tau_truthMatched']
):
    for pt, eta, nTracks, phi, charge, E, truth in zip(pt_list, eta_list, nTracks_list, phi_list, charge_list, E_list, truth_list):
        if not (np.isfinite(pt) and np.isfinite(eta) and np.isfinite(nTracks) and np.isfinite(phi) and np.isfinite(charge) and np.isfinite(E)):
            continue

        X_list.append([pt, eta, nTracks, phi, charge, E])
        y_list.append(int(truth))



#Convert X and y into numpy arrays
X = np.array(X_list, dtype=float)
y = np.array(y_list, dtype=int)   #y = correcttau vs jet structure answers (truth)

print('X shape:', X.shape)
print('y shape:', y.shape)


#Clean the data (remove bad values)
mask = np.isfinite(X).all(axis=1)
X = X[mask]
y = y[mask]


#Split data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

#Class weights - misclassifying a tau is more expensive than missclassifying a background
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print('Class weights:', class_weights)


#Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Build the neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

model = Sequential ([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


#Tell the model how to learn
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


#Train the network
early_stop = EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights = True
)

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=512,
    validation_split=0.2,
    callbacks = [early_stop],
    verbose = 1,
    class_weight=class_weights
)


#Evaluate on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')
print(f'Test loss: {loss:.3f}')

#model output (tau probability)
tau_prob = model.predict(X_test).flatten()
tau_pred = (tau_prob > 0.5).astype(int)

#ROC curve
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, tau_prob)
roc_auc = auc(fpr, tpr)

print(f'ROC AUC: {roc_auc:.3f}')

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Monte Carlo Test Set')
plt.legend()
plt.show()



###################################################
####### RUNNING THE TAU ID ON DETECTOR DATA #######
###################################################


#load detector data
data_filepath = '/home/midway/CERN Open Data/1lep1tau/Data/data_A.1lep1tau.root'
data_file = uproot.open(data_filepath)
data_tree = data_file['mini']

data_features = [
    'tau_pt',
    'tau_eta',
    'tau_nTracks',
    'tau_phi',
    'tau_charge',
    'tau_E'
]

data_arrays = data_tree.arrays(data_features, library='ak')


#Flatten tau candidates
X_data_list = []

for pt_list, eta_list, nTracks_list, phi_list, charge_list, E_list in zip(
    data_arrays['tau_pt'],
    data_arrays['tau_eta'],
    data_arrays['tau_nTracks'],
    data_arrays['tau_phi'],
    data_arrays['tau_charge'],
    data_arrays['tau_E']
):
    for pt, eta, nTracks, phi, charge, E in zip(pt_list, eta_list, nTracks_list, phi_list, charge_list, E_list):
        if not (np.isfinite(pt) and np.isfinite(eta) and np.isfinite(nTracks) and np.isfinite(phi) and np.isfinite(charge) and np.isfinite(E)):
            continue
    
        X_data_list.append([pt, eta, nTracks, phi, charge, E])

X_data = np.array(X_data_list, dtype=float)

print('Detector data shape:', X_data.shape)


#apply same scaler
X_data_scaled = scaler.transform(X_data)

#run the trained model on detector data
tau_scores = model.predict(X_data_scaled).flatten()
print(tau_scores)
