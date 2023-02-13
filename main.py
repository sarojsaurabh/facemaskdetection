import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from keras.utils import load_img,img_to_array
import tempfile

st.set_page_config(page_title="Face Mask Detection",page_icon="https://www.cvisionlab.com/wp-content/themes/cvisionlab/images/editor/cases/mask-detection/icon-mask-detection.png")


st.title("Face Mask Detection System")
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model("mask.h5")
st.sidebar.image("https://hackernoon.com/images/oO6rUouOWRYlzw88QM9pb0KyMIJ3-82bk3j2w.png")
choice=st.sidebar.selectbox("My Menu",("Home","Image","Video","URL"))

if(choice=="Home"):
    st.image("http://via-vis.com/wp-content/uploads/2020/08/Mask-Detection-gif-1.gif")
    st.write("Face Mask Detection System is a Computer vision deep learning application which can be used on detecting mask on perople faces in video as well as in the realtime Footage.")
elif(choice=="Image"):
    file=st.file_uploader("Upload an Image")
    if file:
        b=file.getvalue()                      #converting file into bytes
        k=np.frombuffer(b,np.uint8)           #encoding  image data into form of numpy array 
        img=cv2.imdecode(k,cv2.IMREAD_COLOR)   #decoding
        face=facemodel.detectMultiScale(img)
        for(x,y,l,w) in face:
            face_img=img[y:y+w,x:x+l]          #cropping the face
            cv2.imwrite('temp.jpg',face_img)           #saving the image into local directory        
            face_img=load_img('temp.jpg',target_size=(150,150,3))
            face_img=img_to_array(face_img)            #Converting the image into array
            face_img=np.expand_dims(face_img,axis=0)
            pred=maskmodel.predict(face_img)[0][0]
            if pred==1:
                 cv2.rectangle(img,(x,y),(x+l,y+w),(0,0,255),4)
            else:
                 cv2.rectangle(img,(x,y),(x+l,y+w),(0,255,0),4)
        st.image(img,channels='BGR',width=300)

elif(choice=='Video'):
    file=st.file_uploader('upload a video')
    window=st.empty()
    if file:
        tfile=tempfile.NamedTemporaryFile()
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)
        while(vid.isOpened()):  
            flag,frame=vid.read()
            if flag:
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face:
                    face_img=frame[y:y+w,x:x+l]  #cropping the face
                    cv2.imwrite('temp.jpg',face_img)  #Saving the image into local directory
                    face_img=load_img('temp.jpg',target_size=(150,150,3)) #load the image into a particular target size
                    face_img=img_to_array(face_img) #Converting the image into array
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=maskmodel.predict(face_img)[0][0]     
                    if pred==1:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)

            window.image(frame,channels='BGR')

elif(choice=='URL'):
    link=st.text_input('Enter a link here:')
    btn=st.button('Start')
    window=st.empty()
    if btn:
        
        vid=cv2.VideoCapture(link)
        btn2=st.button("Stop")
        if btn2:
            vid.close()
            st.experimental_rerun()
        while(vid.isOpened()):  
            flag,frame=vid.read()
            if flag:
                face=facemodel.detectMultiScale(frame)
                for (x,y,l,w) in face:
                    face_img=frame[y:y+w,x:x+l]  #cropping the face
                    cv2.imwrite('temp.jpg',face_img)  #Saving the image into local directory
                    face_img=load_img('temp.jpg',target_size=(150,150,3)) #load the image into a particular target size
                    face_img=img_to_array(face_img) #Converting the image into array
                    face_img=np.expand_dims(face_img,axis=0)
                    pred=maskmodel.predict(face_img)[0][0]     
                    if pred==1:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),4)
                    else:
                        cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),4)

            window.image(frame,channels='BGR')             

        
        
        
