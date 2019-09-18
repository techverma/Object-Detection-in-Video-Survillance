import tensorflow as tf
import numpy as np
import cv2
import sys
import Image


tf.app.flags.DEFINE_string("logdir","logs","Directory for saving the logs of computation")
tf.app.flags.DEFINE_string("bgs","MOG2","Algorithm to be used for background substraction. 'MOG' - for MOG; 'MOG2' - for MOG2; 'GMG' - for GMG; 'TM' - for basic temporal median substraction")
tf.app.flags.DEFINE_string("vpath","data/datasample1.mov","Path of the video to be processed")
tf.app.flags.DEFINE_float("resize_ratio",0.25,"Ratio by which each frame should be resized to reduce memory usage")


FLAGS = tf.app.flags.FLAGS


def get_frames(vpath,resize_ratio):
    vpath = FLAGS.vpath
    # resize_ratio =1

    cap = cv2.VideoCapture(vpath)

    if not cap.isOpened():
        print "could not open : ",vpath
        return

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*resize_ratio)
    height = int(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))*resize_ratio)
    fps    = cap.get(cv2.CAP_PROP_FPS)

    print "Number of input frames: ",length
    print "Width: " ,width
    print "Height: " ,height
    print "Frames per second: ", fps

    frames = np.zeros((length,height,width, 3), dtype=np.int64)

    for i in range(0,length):
        ret,frame = cap.read()
        try:
            # print ret,frame
            if ret:
                rframe = cv2.resize(frame,None,fx=resize_ratio,fy=resize_ratio, interpolation=cv2.INTER_CUBIC)
                # print rframe.shape
                # rframe = cv2.cvtColor(rframe,cv2.COLOR_BGR2RGB)
                frames[i] = rframe
                # if i == 500:
                    # Image.fromarray(rframe).save("frame.jpg")
        except Exception as e:
            print "Cannot read video file", e
            # print i,ret
            break

    return frames

def gen_background(frames):
    n,h,w,_ = frames.shape

    bg = np.zeros((h,w,3),dtype=np.uint8)

    for i in range(0,h):
        for j in range(0,w):
            bg[i,j] = np.median(frames[:,i,j,:],axis=0) #temporal median

    Image.fromarray(cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)).save('images/bg.jpg');

    return bg

def bg_separation(frames,t):

    if(FLAGS.bgs == "MOG"):
        separator = cv2.createBackgroundSubtractorMOG()
    elif (FLAGS.bgs == "MOG2"):
        separator = cv2.createBackgroundSubtractorMOG2()
    elif (FLAGS.bgs == "GMG"):
        separator = cv2.createBackgroundSubtractorGMG()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    else:
        print FLAGS.bgs,"method not supported"
        sys.exit(0)

    n,h,w,_ = frames.shape
    output_frames = np.zeros((n,h,w),dtype=np.uint8)

    for i in range(n):
        output_frames[i] = separator.apply(frames[i])


    return output_frames

def gen_activity(frames):
    k,h,w = frames.shape
    video = cv2.VideoWriter('activity_video_synopsis.mp4',cv2.VideoWriter_fourcc(*'X264'),10,(w,h),False)

    for i in range(0,k):
        video.write(frames[i]*255)


    cv2.destroyAllWindows()
    video.release()

    return 1;

def main(vpath,resize_ratio):

    graph = tf.Graph()

    with graph.as_default():

        with tf.name_scope("input") as scope:
            # video_path = tf.constant(vpath)
            video_path = tf.constant(0,name="video_path") #commented cause tf.string is not supported yet
            rRatio = tf.constant(resize_ratio, dtype=tf.float32,name="resize_ratio")
            activity_tolerance = tf.constant(1000, dtype=tf.int32, name="tolerance")

        with tf.name_scope("video_processing") as scope:
            frames = tf.py_func(get_frames,[video_path,rRatio],[tf.int64],name="get_frames")[0]

        with tf.name_scope("background_separation") as scope:
            background = tf.py_func(gen_background,[frames],[tf.uint8],name="background")[0]
            activity_matrix = tf.py_func(bg_separation,[frames],[tf.uint8],name="activity_matrix")[0]
            temporal_activity_matrix = tf.cast(tf.greater(tf.reduce_sum(tf.square((tf.to_int32(frames)-tf.to_int32(background))),3),activity_tolerance),tf.uint8,name="temp_activity_matrix")
            activity_video = tf.py_func(gen_activity,[activity_matrix],[tf.int64],name="activity_video")[0]
            temp_activity_video = tf.py_func(gen_activity,[temporal_activity_matrix],[tf.int64],name="activity_video")[0]

        sess = tf.Session()
        writer = tf.train.SummaryWriter(FLAGS.logdir,sess.graph_def)

        fs,tavideo = sess.run([frames,temp_activity_video])

        return fs

fs = main(FLAGS.vpath,FLAGS.resize_ratio)
print fs.shape
