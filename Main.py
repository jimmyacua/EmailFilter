from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Input, Dropout, Dense, LeakyReLU
from keras.callbacks import ModelCheckpoint
from sklearn.decomposition import PCA


#CLASS 0 = Personal, CLASS 1 = GUBERNAMENTAL

df = pd.read_csv("Emails_clean.csv")

#print(df.describe())

#print(len(df))

df.convert_objects(convert_numeric= True)
df.fillna(0, inplace=True)


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elemts = set(column_contents)
            x = 0
            for unique in unique_elemts:
                if unique not in text_digit_vals:
                    if unique == '0':
                        text_digit_vals[unique] = 0
                    else:
                        text_digit_vals[unique] = x+1
                        x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)
#print(df)
#train, test = train_test_split(df, test_size=0.1)




#dropping the solution
pca_data = df.drop("Class", 1)

pca = PCA(n_components=5)
pca.fit(pca_data)

pca_data = pd.DataFrame(pca.transform(pca_data))
#print(pca_data.shape)


#normalizar datos
for col in range(pca_data.shape[1]):
    mn = np.mean(pca_data.iloc[:, col])
    st = np.mean(pca_data.iloc[:, col].std())

    pca_data.iloc[:, col] = (pca_data.iloc[:, col] - mn) / st
    pca_data.iloc[:, col] = np.nan_to_num(pca_data.iloc[:, col])

#print(pca_data)


test_ratio=0.1 #10% de los datos

new_data = pd.concat([pca_data, df["Class"]],1)

test_data = new_data.sample(frac=test_ratio)
train_data = new_data.drop(test_data.index)

test_sols = test_data["Class"]
test_data = test_data.drop("Class", 1)

train_sols = train_data["Class"]
train_data = train_data.drop("Class", 1)

#print(train_data)


train_sols = pd.get_dummies(train_sols, prefix="Class")
test_sols = pd.get_dummies(test_sols, prefix="Class")




inp = Input(shape=(5,))
x = Dense(128)(inp)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(64)(inp)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(32)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(8)(x)
x = LeakyReLU()(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)

model = Model(inputs=inp, outputs=x)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

weight0 = 1.0/train_sols["Class_0"].sum()
weight1 = 1.0/train_sols["Class_1"].sum()

_sum = weight0+weight1

weight0 /= _sum
weight1 /= _sum

#print("weight0", weight0)
#print("weight1", weight1)

callback = [ModelCheckpoint("results.h5", save_best_only=True, monitor="val_acc", verbose=0)]

model.fit(train_data, train_sols, batch_size=500, epochs=150, verbose=0, callbacks=callback, validation_split=0.2, shuffle=True,
         class_weight={0:weight0, 1: weight1})

best_model = load_model("results.h5")

score = model.evaluate(test_data, test_sols)
print("Porcentaje de aciertos: ", score[1])


