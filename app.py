from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template ,url_for,redirect
from werkzeug.utils import secure_filename
import sqlite3
import cv2
import csv

import os, sys, glob, re

app = Flask(__name__)

model_path = "model002.h5"

SoilNet = load_model(model_path)

classes = {0:"Bacterial_Blight",1:"Brown_Spots",2:"Brown_Stripe",3:"Healthy",4:"Late_Wilt",5:"Leaf_Smut"}

def resize_images(img):
  img = np.array(img).astype(np.uint8)
  res = cv2.resize(img,(224,224), interpolation = cv2.INTER_CUBIC)
  return res
#image = [resize_images(img) for img in image]

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp

def segment(img):
  image_sharpen = sharpen_image(img)
  return image_sharpen

def model_predict(image_path,model):
    print("Predicted")
    image = load_img(image_path,target_size=(224,224))
    image=resize_images(image)
    image = img_to_array(image)
    image=segment(image)
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    print(result)
    prediction = classes[result]
    
    
    
    if result == 0:
        
        
        return "Bacterial_Blight","BacterialLeafBlight.html"
    elif result == 1:
        
        
        return "Brown_Spots", "BrownSpotInWheat.html"
    elif result == 2:
       
        
        return "Brown_Stripe" , "BrownStripeOfMaize.html"
    
    elif result==3:
        return "Healthy", "Healthy.html"
    
    elif result==4:
        return "Late_Wilt", "LateWiltofMaize.html"
    
    elif result==5:
        return "Leaf_Smut_Wheat", "LeafSmut.html"
    
    
@app.route("/")
def home():
    # return the homepage
    return render_template("login.html")

@app.route("/signup")
def signup():
    name = request.args.get('username','')
    dob = request.args.get('DOB','')
    sex = request.args.get('Sex','')
    contactno = request.args.get('CN','')
    email = request.args.get('email','')
    martial = request.args.get('martial','')
    password = request.args.get('psw','')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `accounts` (`name`, `dob`,`sex`,`contact`,`email`,`martial`, `password`) VALUES (?, ?, ?, ?, ?, ?, ?)",(name,dob,sex,contactno,email,martial,password))
    con.commit()
    con.close()

    return render_template("login.html")

@app.route("/signin")
def signin():
    mail1 = request.args.get('uname','')
    password1 = request.args.get('psw','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `email`, `password` from accounts where `email` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("login.html")

    elif mail1 == data[0] and password1 == data[1]:
        return render_template("index.html")

    
    else:
        return render_template("login.html")


@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/index',methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    print("Entered")
    if request.method == 'POST':
        print("Entered here")
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = model_predict(file_path,SoilNet)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    


if __name__ == '__main__':
    app.run(debug=False,threaded=False)
    
