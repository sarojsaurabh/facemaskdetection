from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

#Define the model
mymodel=Sequential()
mymodel.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Conv2D(32,(3,3),activation='relu'))
mymodel.add(MaxPooling2D())
mymodel.add(Flatten())
mymodel.add(Dense(100,activation='relu'))
mymodel.add(Dense(1,activation='sigmoid'))
mymodel.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['Accuracy'])

#Organize the image data
train=ImageDataGenerator(rescale=1./255,zoom_range=0.2,horizontal_flip=True)
test=ImageDataGenerator(rescale=1./255)
train_img=train.flow_from_directory('train',target_size=(150,150),batch_size=16,class_mode='binary')
test_img=train.flow_from_directory('test',target_size=(150,150),batch_size=16,class_mode='binary')

#Train and Test the model
maskmodel=mymodel.fit(train_img,epochs=10,validation_data=test_img)

#Save the model
mymodel.save("mask.h5",maskmodel)
