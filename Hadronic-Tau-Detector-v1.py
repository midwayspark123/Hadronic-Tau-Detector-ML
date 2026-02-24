import uproot
import numpy as np
import awkward as ak

#Loading the root file
filepath = 'filepath'
file = uproot.open(filepath)

print(file.keys())
tree = file['mini']

#Choosing the variables to use in the comparison
features = [
    'tau_pt', #tau_pt = energy scale
    'tau_eta', #tau_eta = detector location
    'tau_nTracks', #tau_nTracks = tau vs jet structure
    'tau_BDTid' #tau_BDTid = ATLAS tau ID score
]

#Load tau branches into awkward arrays
arrays = tree.arrays(features + ["tau_truthMatched"], library='ak')

#create lists to store per-tau candidate
X_list = []
y_list = []

for pt_list, eta_list, nTracks_list, BDTid_list, truth_list in zip(
    arrays['tau_pt'], arrays['tau_eta'], arrays['tau_nTracks'], arrays['tau_BDTid'], arrays['tau_truthMatched']
):
    for pt, eta, nTracks, BDTid, truth in zip(pt_list, eta_list, nTracks_list, BDTid_list, truth_list):
        if not (np.isfinite(pt) and np.isfinite(eta) and np.isfinite(nTracks) and np.isfinite(BDTid)):
            continue

        X_list.append([pt, eta, nTracks, BDTid])
        y_list.append(int(truth))



#Convert X and y into numpy arrays
X = np.array(X_list, dtype=float)
y = np.array(y_list, dtype=int)   #y = correct answers (truth)

print('X shape:', X.shape)
print('y shape:', y.shape)


#stack the flattened jagged arrays into a numpy array

#Clean the data (remove bad values)
mask = np.isfinite(X).all(axis=1)
X = X[mask]
y = y[mask]


#Split data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)


#Build the neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential ([
    Dense(16, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])


#Tell the model how to learn
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


#Train the network
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2
)


#Test the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)

#Get tau probabilites
tau_prob = model.predict(X_test)

#Convert probabilities to decisions
tau_prediction = (tau_prob >= 0.5).astype(int)
print(tau_prediction)
