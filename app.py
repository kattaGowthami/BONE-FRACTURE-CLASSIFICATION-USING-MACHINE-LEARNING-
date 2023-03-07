from flask import Flask, request,flash,render_template,send_from_directory,redirect,url_for
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras.models import load_model
import numpy as np
import mysql.connector
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import shutil

data_dir = 'Data'
Bone = []
for file in os.listdir(data_dir):
    Bone += [file]
print(Bone)

app=Flask(__name__)
app.secret_key='random string'



mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    port='3307',
    database="bone"
)
mycursor = mydb.cursor()


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/reg')
def reg():
    return render_template("reg.html")

@app.route('/regback',methods = ["POST"])
def regback():
    if request.method=='POST':
        name=request.form['name']
        email=request.form['email']
        pwd=request.form['pwd']
        cpwd=request.form['cpwd']
        addr=request.form['addr']

        sql = "select * from ureg"
        result = pd.read_sql_query(sql, mydb)
        email1 = result['email'].values
        print(email1)
        if email in email1:
            flash("email already existed", "success")
            return render_template('index.html')
        if (pwd == cpwd):
            sql = "INSERT INTO ureg (name,email,pwd,addr) VALUES (%s,%s,%s,%s)"
            val = (name, email, pwd, addr)
            mycursor.execute(sql, val)
            mydb.commit()
            flash("Successfully Registered", "warning")
            return render_template('index.html')
        else:
            flash("Password and Confirm Password not same")

    return render_template('index.html')

@app.route('/login')
def login():
    return render_template("login.html")


@app.route('/logback',methods=['POST', 'GET'])
def logback():
    if request.method == "POST":

        email = request.form['email']

        password1 = request.form['pwd']
        print('p')

        sql = "select * from ureg where email='%s' and pwd='%s' " % (email, password1)
        print('q')
        x = mycursor.execute(sql)
        print(x)
        results = mycursor.fetchall()
        print(results)
        global name

        if len(results) > 0:

            flash("Welcome ", "primary")
            return render_template('userhome.html', msg=results[0][1])
        else:
            flash("Invalid Credentials ", "danger")
            return render_template('index.html')

    return render_template('index.html')

@app.route('/userhome')
def userhome():
    return render_template("userhome.html")

@app.route('/upload')
def upload():
    print('sssssssssssssssssssss')
    return render_template("upload.html")

@app.route('/upload1/<filename>')
def send_image(filename):
    print('kjsifhuissywudhj')
    return send_from_directory("images", filename)

@app.route("/upload1", methods=["POST","GET"])
def upload1():
    print('a')
    if request.method == 'POST':
        print("hdgkj")
        m = int(request.form["alg"])
        acc=pd.read_csv("ACC.csv")
        myfile = request.files['file']
        fn = myfile.filename
        mypath = os.path.join("images/", fn)
        myfile.save(mypath)

        print("{} is the file name", fn)
        print("Accept incoming file:", fn)
        print("Save it to:", mypath)

        if m == 1:
            print("bv1")
            new_model = load_model('model/svm.h5')
            test_image = image.load_img(mypath, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image/=255
            a=acc.iloc[m-1,1]
        else:
            print("bv2")
            new_model = load_model('model/cnn.h5')
            test_image = image.load_img(mypath, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image /= 255
            a = acc.iloc[m - 1, 1]


        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        preds = Bone[np.argmax(result)]

        if preds=="Fracture":
            msg="Exercies: Weight-bearing exercises like walking, climbing stairs, dancing and playing tennis are good for increasing bone strength, and resistance training — such as lifting weights — builds muscle to support your bones.|Food: Red meat, dark-meat chicken or turkey, oily fish, eggs, dried fruits, leafy green veggies, whole-grain breads, and fortified cereals"
        else:
            msg="Not Fracture"
        msg = msg.split('|')
        print(msg)

    return render_template("template.html", text=preds,  msg=msg,image_name=fn,a=round(a*100,3))


if __name__=='__main__':
    app.run(debug=True)