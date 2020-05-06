import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from tensors import to_tensor_input

df = pd.read_csv('core/output/batters.csv')
indexer = df.reset_index()[['index', 'retroID']].to_dict()['retroID']
y = df['Batting'].values
to_drop = ['debutYear', 'finalYear', 'G', '1B', 'AB', 'RBI', 'wOBA']
df.drop(columns=to_drop, inplace=True)


X = df.drop(columns=['Batting']).values
y = df[['retroID', 'Batting']].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)
X_train_keys = np.asarray([x[0] for x in X_train])
X_train = np.asarray([x[1:] for x in X_train])
X_test_keys = np.asarray([x[0] for x in X_test])
X_test = np.asarray([x[1:] for x in X_test])
y_train_keys = np.asarray([y[0] for y in y_train])
y_train = np.asarray([y[1] for y in y_train])
y_test_keys = np.asarray([y[0] for y in y_test])
y_test = np.asarray([y[1] for y in y_test])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# def to_tensor_input(player):
#     return scaler.transform(player.values.reshape(-1, 29))[0]


tensor = df.drop(columns=['retroID', 'Batting'])
player_tensor_inputs = tensor.apply(
    lambda player: to_tensor_input(scaler, player, 'batting'), axis=1)

tensor = pd.DataFrame(player_tensor_inputs.values.tolist())

tensor.to_csv('core/tensors/t_batting.csv', index=False, float_format='%g')

epochs = 400
batch_size = 64
loss_param = 'mse'
optimizer_param = 'adam'
stop_monitor = 'val_loss'
stop_patience = 20

model = Sequential()

model.add(Dense(116, activation='relu', kernel_regularizer=regularizers.l2(
    0.0001), input_shape=X_train.shape[1:]))

model.add(Dropout(0.5))

model.add(Dense(232, activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dropout(0.5))

model.add(Dense(116, activation='relu',
                kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss=loss_param, optimizer=optimizer_param)

early_stop = EarlyStopping(monitor=stop_monitor, patience=stop_patience)

results = model.fit(x=X_train, y=y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop]
                    )

losses = model.history.history
losses['loss'] = np.asarray(losses['loss'])
losses['val_loss'] = np.asarray(losses['val_loss'])
final_number_of_epochs = len(losses['loss'])
min_loss = losses['loss'].min()
mean_loss = losses['loss'].mean()
final_loss = losses['loss'][-1]
min_val_loss = losses['val_loss'].min()
mean_val_loss = losses['val_loss'].mean()
final_val_loss = losses['val_loss'][-1]


def get_model_summary():
    output = []
    model.summary(print_fn=lambda line: output.append(line))
    return str(output).strip('[]')


summary = get_model_summary()

record = {
    'Epochs': final_number_of_epochs,
    'Batch_Size': batch_size,
    'Loss_Func': loss_param,
    'Optimizer': optimizer_param,
    'Early_Stop_Monitor': stop_monitor,
    'Early_Stop_Patience': stop_patience,
    'Min_Loss': min_loss,
    'Mean_Loss': mean_loss,
    'Final_Loss': final_loss,
    'Min_Val_Loss': min_val_loss,
    'Mean_Val_Loss': mean_val_loss,
    'Final_Val_Loss': final_val_loss,
    'Model': summary
}

new_data = pd.DataFrame(record, index=[0])

if os.path.exists('core/records/batting_results.csv'):
    df_records = pd.read_csv('core/records/batting_results.csv')
    df_records.append(new_data)
else:
    df_records = pd.DataFrame(new_data)

df_records.to_csv('core/records/batting_results.csv',
                  index=False, float_format='%g')

model.save('core/models/model_batting.h5')
