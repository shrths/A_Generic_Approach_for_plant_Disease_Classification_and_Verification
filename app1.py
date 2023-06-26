from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template ,url_for,redirect
from werkzeug.utils import secure_filename
import sqlite3
import cv2
import csv
import pickle

import os, sys, glob, re

app = Flask(__name__)

model_path = "model_final1_3.h5"

SoilNet = load_model(model_path)

classes = {0:"Bacterial_Blight_in_Rice",1:"Bacterial_Leaf_Blight_Wheat",2:"Bacterial_Leaf_Spot_Ragi",3:"Brown_Spot_in_Rice",4:"Brown_Spot_Wheat",5:"Brown_Stripe_Of_Maize",6:"Foot_Wilt_Ragi",7:"Healthy_Maize",8:"Healthy_Ragi",9:"Healthy_Rice",10:"Healthy_Wheat",11:"Late_Wilt_of_Maize",12:"Leaf_Smut_Wheat"}

#classes = {0:"Brown_Spot_Wheat",1:"Healthy_Wheat",2:"Bacterial_Blight_in_Rice",3:"Brown_Spot_in_Rice",4:"Foot_Wilt_Ragi",5:"Healthy_Rice",6:"Healthy_Ragi",7:"Leaf_Smut_Wheat" ,8:"Bacterial_Leaf_Blight_Wheat", 9:"Brown_Stripe_of_Maize" ,10:"Healthy_Maize" ,11:"Bacterial_Leaf_Spot_Ragi",12:"Late_Wilt_of_Maize" }
#with open(os.path.join("C:\Projectcode\A Generic Approach for Wheat AND RICE Disease Classification and Verification Using  Expert Opinion for Knowledge-Based Decisions","labels_list.pkl"),"rb") as handle:
    #labels_id = pickle.load(handle)

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
    
    #image = image/255
    image = np.expand_dims(image,axis=0)
    
    result = np.argmax(model.predict(image))
    #id_labels = dict()
    #for class_name,idx in labels_id.items():
      #  id_labels[idx] = class_name
    #print(id_labels)
    #ypred_class = int(ypred_class)
    #res = id_labels[result]
   # prediction =res
    
    prediction = classes[result]
    print(result)
    print(prediction)
    
    
    
    if prediction == 0:
        return "Brown_Spot_Wheat", "BrownSpotInWheat.html"
        

    elif prediction == 1:
        return "Healthy_Ragi" , "Healthy.html"
       
    elif prediction == 2:
        return "Leaf_Smut_Wheat", "LeafSmut.html"

        
    
    elif prediction == 3:
        return "Brown_Spot_in_Rice","BrownSpotInRice.html"
        
    
    elif prediction == 4:
        return "Healthy_Wheat", "Healthy.html"

    elif prediction == 5:
        return "Late_Wilt_of_Maize", "LateWiltofMaize.html"
        

    elif prediction == 6:
        return "Bacterial_leaf_Spot_Ragi","BacterialLeafSpot.html"
    
    elif prediction == 7:
        return "Healthy_Maize" , "Healthy.html"

    if prediction == 8:
        return "Brown_Stripe_of_Maize" , "BrownStripeOfMaize.html"

    elif prediction == 9:
        return "Foot_Wilt_Ragi" , "FootWilt.html"

    elif prediction == 10:
        return "Bacterial_Leaf_Blight_Wheat" , "BacterialLeafBlight.html"
    
    elif prediction == 11:
        return "Bacterial_Blight_in_Rice", "BacterialBlightInRice.html"
    
    elif prediction == 12:
        return "Healthy_Rice" , "Healthy.html"
    
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
    
