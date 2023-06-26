from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template ,url_for,redirect
from werkzeug.utils import secure_filename
import sqlite3
import cv2


import os, sys, glob, re

app = Flask(__name__)

model_path = "model_final1_3.h5"

SoilNet = load_model(model_path)

labels = ['Bacterial_Blight_in_Rice', 'Bacterial_Leaf_Blight_Wheat',
       'Bacterial_Leaf_Spot_Ragi', 'Brown_Spot_Wheat',
       'Brown_Spot_in_Rice', 'Brown_Stripe_of_Maize', 'Foot_Wilt_Ragi',
       'Healthy_Maize', 'Healthy_Ragi', 'Healthy_Rice', 'Healthy_Wheat',
       'Late_Wilt_of_Maize', 'Leaf_Smut_Wheat']



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
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = resize_images(image_bgr)
    img = segment(img)
    # print(img.shape)
    imgs = []
    imgs.append(img)
    imgs = np.array(imgs)
    a = model.predict(imgs)
    result = np.argmax(a,axis = 1) 
    return labels[result[0]]
    
    
    
    
    
    
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
        pred = model_predict(file_path,SoilNet)

        if pred == "Bacterial_Blight_in_Rice":
            output_page = "BacterialBlightInRice.html"
        
        
        elif pred == "Bacterial_Leaf_Blight_Wheat":
            output_page = "BacterialLeafBlight.html"
            
        
        elif pred == "Bacterial_Leaf_Spot_Ragi":
            output_page = "BacterialLeafSpot.html"
            

        elif pred == "Brown_Spot_Wheat":
            output_page ="BrownSpotInWheat.html"
            
        
        elif pred == "Brown_Spot_in_Rice":
            output_page = "BrownSpotInRice.html"
            

        elif pred == "Brown_Stripe_of_Maize":
            output_page  = "BrownStripeOfMaize.html"
            

        elif pred == "Foot_Wilt_Ragi":
            output_page  = "FootWilt.html"
        
        elif pred == "Healthy_Maize":
            output_page  = "Healthy.html"

        if pred == "Healthy_Ragi":
            output_page  = "Healthy.html"

        elif pred == "Healthy_Rice":
            output_page  = "Healthy.html"

        elif pred == "Healthy_Wheat":
            output_page = "Healthy.html"
        
        elif pred == "Late_Wilt_of_Maize":
            output_page = "LateWiltofMaize.html"
        
        elif pred == "Leaf_Smut_Wheat":
            output_page = "LeafSmut.html"
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    


if __name__ == '__main__':
    app.run(debug=False,threaded=False)
    
