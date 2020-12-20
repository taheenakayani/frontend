from __future__ import division, print_function, absolute_import
from timeit import time
import warnings
from datetime import datetime as dt
warnings.filterwarnings('ignore')
from flask import Flask, render_template, Response, request,send_from_directory,jsonify,send_file
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
import cv2
from bson.objectid import ObjectId
import numpy as np
import shelve
from PIL import Image
from werkzeug.utils import secure_filename
from yolo3.yolo import YOLO
from tools import processing
from tools import generate_detections as gdet
from flask_cors import CORS, cross_origin
from tools.processing import extract_parts
from tools.coord_in_box import coordinates_in_box,bbox_to_fig_ratio
import os
from flask_pymongo import PyMongo
from deepsort import nn_matching
from deepsort.detection import Detection
from deepsort.tracker import Tracker
from models.openpose_model import pose_detection_model
from config.config_reader import config_reader
UPLOAD_FOLDER = ''
from training.data_preprocessing import batch,generate_angles
from keras.models import load_model
app = Flask(__name__)
app.config['DEBUG']=False
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER']=''
app.config['MONGO_DBNAME'] = 'fyp'
# app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['MONGO_URI'] = 'mongodb+srv://usman:ros3BrHGDfJR8cFG@cluster0.qflg8.mongodb.net/fyp?retryWrites=true&w=majority'
app.config["JWT_SECRET_KEY"] = "roman"
jwt = JWTManager(app)
mongo = PyMongo(app)
cors = CORS(app)
yolo = YOLO()
user=mongo.db.admins
# Intializing OpenPose Model
model = pose_detection_model()

# Defining parameters for openpose model
param, model_params = config_reader()

# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# Deep SORT
model_filename = 'models/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)

# Initializing the tracker with given metrics.
tracker = Tracker(metric)


model_ts = load_model('./models/Time Series.h5')



@app.route("/register", methods=["POST"])
@cross_origin()
def register():
    email = request.json["email"]
    # test = User.query.filter_by(email=email).first()
    test = user.find_one({"email": email})
    if test:
        return jsonify(message="User Already Exist"), 409
    else:
        username = request.json["username"]
        password = request.json["password"]
        user_info = dict(username=username, email=email, password=password)
        user.insert_one(user_info)
        return jsonify(message="User added sucessfully"), 201


@app.route("/login", methods=["POST"])
@cross_origin()
def login():
    if request.is_json:
        username = request.json["username"]
        password = request.json["password"]
    else:
        username = request.form["username"]
        password = request.form["password"]

    test = user.find_one({"username": username, "password": password})
    if test:
        access_token = create_access_token(identity=username)
        return jsonify(message="Login Succeeded!", access_token=access_token), 201
    else:
        return jsonify(message="Bad Username or Password"), 401




@app.route('/imagestar/<imgname>',methods=['GET'])
@cross_origin()
def image_star(imgname):
    return send_file(imgname, mimetype='image/gif')  






@app.route('/retrieve', methods=['POST'])
@cross_origin()
def get_all_stars():
    star = mongo.db.records
    output = []
    for s in star.find({"cid":str(request.json["data"])}):
        sid=str(s['_id'])
        output.append({'culprits': s['culprits'], 'video': s['video'],'date':s['date'],'time':s['time'],'_id':sid,'image':s['image'] ,'cid':s['cid']})
    return jsonify({'result': output}) 


@app.route('/delstar/<pid>', methods=['DELETE'])
@cross_origin()
def del_star(pid):
    output = []
    star = mongo.db.records
    print(pid)
    newid=ObjectId(pid)
    result = star.find_one_and_delete({'_id':newid})
    if(result):
       for s in star.find():
        sid=str(s['_id'])
        output.append({'culprits': s['culprits'], 'video': s['video'],'date':s['date'],'time':s['time'],'_id':sid,'image':s['image'] ,'cid':s['cid']})
    return jsonify({'result': output})
    # print(result)
    # return jsonify({'result': "done"})  



@app.route('/uploads/<filename>')
@cross_origin()
def uploaded_file(filename):
    """Endpoint to serve uploaded videos

    Use `conditional=True` in order to support range requests necessary for
    seeking videos.

    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename,
                               conditional=True)

@app.route('/countcameras')
def getcount():
    result=countCameras()
    return jsonify(result)


def clearCapture(capture):
    capture.release()
    cv2.destroyAllWindows()

def countCameras():
    n = 0
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            ret, frame = cap.read()
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clearCapture(cap)
            n += 1
        except:
            clearCapture(cap)
            break
    return n


@app.route('/upload', methods=['POST'])
@cross_origin()
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER,'test_docs')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    # session['uploadFilePath']=destination
    response="done uploading"
    return response

def dumbed():
  return "done"

@app.route('/video_feed', methods=['GET', 'POST'])
def video_feed():
    signal =request.args.get('s')
    signal=int(signal)
    return Response(process(signal),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



def process(signal):
# Intializing YOLO model
  writeVideo_flag=True 
  video_capture=cv2.VideoCapture(signal-1) 
  basename = "video"  #changing paths
  suffix = dt.now().strftime("%y%m%d_%H%M%S")
  # e.g. 'mylogfile_120508_171442'
  filename = "_".join([basename, suffix])+".mp4"
  if writeVideo_flag:
    w=int(video_capture.get(3))
    h=int(video_capture.get(4))
    # video_capture.set(3,640)
    # video_capture.set(4,480)
    # fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out=cv2.VideoWriter(filename,fourcc,6, (w, h))
  # Define the codec and create VideoWriter object
  frame_index=0
  person_TS={}
  count=0 
  fps=0.0
  labels={}
  check=[]
  culprits=0
  imgnames=[]
  star = mongo.db.records
  create = False
  condition =0 
  while True:
      ret,frame=video_capture.read()  # frame shape 640*480*3
      #print(ret)
    
      if ret!=True:
          break
      if count%20!=0:
   		  print('SKIPPED {} FRAME'.format(count))
   		  count+=1
      else:
          # t1=time.time()
          image=Image.fromarray(frame[...,::-1]) #bgr to rgb
          boxs=yolo.detect_image(image)
          features=encoder(frame,boxs)
          # score to 1.0 here).
          detections=[Detection(bbox,1.0,feature) for bbox,feature in zip(boxs,features)]
          # Run non-maxima suppression.
          boxes=np.array([d.tlwh for d in detections])
          scores=np.array([d.confidence for d in detections])
          indices=processing.non_max_suppression(boxes,nms_max_overlap,scores)
          detections=[detections[i] for i in indices]
        
          # Call the tracker
          tracker.predict()
          tracker.update(detections)
          person_dict=extract_parts(frame,param,model,model_params)
        
          for track in tracker.tracks:
              if not track.is_confirmed() or track.time_since_update>1:
                  continue
              bbox=track.to_tlbr()
            
              flag=0
            
              # Association of tracking with body keypoints
              for i in person_dict.keys():
                  # If given body keypoints lie in the bounding box or not.
                  if coordinates_in_box(bbox,list(person_dict[i].values())) and bbox_to_fig_ratio(bbox,list(person_dict[i].values())):
                      if 'person_'+str(track.track_id) not in person_TS.keys():
                          person_TS['person_'+str(track.track_id)]=[]
                    
                      person_TS['person_'+str(track.track_id)].append(person_dict[i])
                      flag=1
                      break
              if flag==1:
                  del(person_dict[i])
            

              if track.track_id not in labels.keys():
                  labels[track.track_id]=0

              if not labels[track.track_id] and 'person_'+str(track.track_id) in person_TS.keys():                              #If not violent previously
                  if len(person_TS['person_'+str(track.track_id)])>=6:
                      temp=[]
                      for j in person_TS['person_'+str(track.track_id)][-6:]:
                          temp.append(generate_angles(j))
                      angles=batch(temp)
                      target=int(np.round(model_ts.predict(angles)))
                      labels[track.track_id]=target
              # print([person_TS])
              # print("tracker",track.track_id)
              # print("labels",labels)
              if(len(labels)>len(check)):
                  length=int((len(labels)-len(check)))
                  check.extend(False for i in range(length))
              if labels[track.track_id]:
                  if not (check[track.track_id-1]):
                    culprits+=1
                    x = int(bbox[0])
                    y = int(bbox[1])
                    w = int(bbox[2])
                    h = int(bbox[3])
                    crop_img = frame[y:y + h, x:x + w]
                    iname="person {}.jpg".format(track.track_id-1)
                    cv2.imwrite(iname, crop_img)
                    imgnames.append(iname)
                  check[track.track_id-1]=True
                  color=(0,0,255)
              else:
                  color=(0,255,0)
              # print("check",check)  
              cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),color,2)
            
          frame_index+=1
          # count+=1

          if writeVideo_flag and 1 in labels.values() and create:
                suffix = dt.now().strftime("%y%m%d_%H%M%S")
                filename = "_".join([basename, suffix])+".mp4"
                out = cv2.VideoWriter(filename, fourcc, 6, (640, 480))
                create = False

          if 1 in labels.values():
                culprits=len(faces)
                out.write(frame)
                condition = 1

          if(condition == 1 and  not 1 in labels.values()):
                out = None
                condition = 0
            
                culprits1=str(culprits)
                video = path+".mp4"
                today=dt.today()
                now = dt.now()
                date = today.strftime("%B %d, %Y")
                time = now.strftime("%H:%M:%S")

                star_id = star.insert({'culprits': culprits1, 'video': video,
                          'date': date, 'time': time, 'image': imgnames,'cid':signal})
  
                create = True

          if (np.shape(frame) != ()):
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')    
        # fps=(fps+(1./(time.time()-t1)))/2
        # print("fps= %f"%(fps))
        # print('PROCESSED {} FRAME'.format(count))
         
	     
  
  video_capture.release()



if __name__ == '__main__':
    app.run(host='0.0.0.0',threaded=False)


# def get_length(input_video):
#     result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     return float(result.stdout)
