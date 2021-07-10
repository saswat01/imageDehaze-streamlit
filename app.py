import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
#import tensorflow_hub as hub
#import tensorflow as tf
import numpy as np
#from tensorflow import keras
#from tensorflow.keras.models import load_model
#from tensorflow.keras import preprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 
from cv2.ximgproc import guidedFilter
#from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.io import imread
fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Image Dehazing')

st.markdown("Welcome to this simple web application for image dehazing.")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Dehaze")
    #st.subheader('Minimum allowed value of transmission')
    #y = st.slider('Choose between 0.1 to 0.9', min_value=0.0, max_value=0.9, step=0.1)
    if file_uploaded is not None:    
        image = imread(file_uploaded) 
        st.image(image, caption='Uploaded Image', use_column_width=True)
        #st.subheader('Minimum allowed value of transmission')
        y = st.slider('Choose between 0.1 to 0.9 for Minimum allowed value of transmission', min_value=0.0, max_value=0.9, step=0.1)

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                dehazedimage = imageDehaze(image, y)
                time.sleep(1)
                st.success('Processed')
                st.write(dehazedimage)
                st.pyplot(fig)


# def predict(image):
#     classifier_model = "base_dir.h5"
#     IMAGE_SHAPE = (224, 224,3)
#     model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
#     test_image = image.resize((224,224))
#     test_image = preprocessing.image.img_to_array(test_image)
#     test_image = test_image / 255.0
#     test_image = np.expand_dims(test_image, axis=0)
#     class_names = [
#           'Backpack',
#           'Briefcase',
#           'Duffle', 
#           'Handbag', 
#           'Purse'] 
#     predictions = model.predict(test_image)
#     scores = tf.nn.softmax(predictions[0])
#     scores = scores.numpy()
#     results = {
#           'Backpack': 0,
#           'Briefcase': 0,
#           'Duffle': 0, 
#           'Handbag': 0, 
#           'Purse': 0
# }

    
#     result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
#     return result

def KNB(mw,K):
    mwg = np.double( (0.2989*mw[:,:,0] + 0.5870*mw[:,:,0] + 0.1140*mw[:,:,0])) # RGB to Gray Conversion
    #mwg = rgb2gray(mw)
    r = np.size(mwg)
    
    re,co = np.shape(mwg)
    nbh = np.zeros([re,co,3])
    #nbh_t = zeros([r,3])
    nbh_v = np.zeros([K,3])
    cent = mwg[int(np.floor(re/2)), int(np.floor(co/2))]
    dist = np.zeros(r)
    dist = abs(mwg - cent)
    dist_ord = np.sort(np.ravel(dist[:]))
    dist_K = dist_ord[K-1]
    x,y = np.where(dist <= dist_K)
    nbh[x,y,0] = mw[x,y,0]
    nbh[x,y,1] = mw[x,y,1]
    nbh[x,y,2] = mw[x,y,2]
    nbh_t0 = sorted(np.ravel(nbh[:,:,0]),reverse=True)
    nbh_t1 = sorted(np.ravel(nbh[:,:,1]),reverse=True)
    nbh_t2 = sorted(np.ravel(nbh[:,:,2]),reverse=True)
    nbh_v[:,0] = nbh_t0[0:K]
    nbh_v[:,1] = nbh_t1[0:K]
    nbh_v[:,2] = nbh_t2[0:K]
    return nbh_v

def imageDehaze(image, y):
    S = 19

    f_c = np.double(image)
    Nr,Nc,Np = f_c.shape 
    # Extention of the input image to avoid edge issues
    A1 = np.concatenate((np.flipud(np.fliplr(f_c)), np.flipud(f_c), np.flipud(np.fliplr(f_c))), axis=1)
    A2 = np.concatenate((np.fliplr(f_c),            f_c,            np.fliplr(f_c)), axis=1)
    A3 = np.concatenate((np.flipud(np.fliplr(f_c)), np.flipud(f_c), np.flipud(np.fliplr(f_c))), axis=1)
    f_proc = np.concatenate( (A1,A2,A3) ,axis=0)
    f_proc = f_proc[Nr-int((S-1)/2):2*Nr+int((S-1)/2), Nc-int((S-1)/2):2*Nc+int((S-1)/2),:]

    A_test = np.zeros([Nr,Nc])
    f_mv = np.zeros([S,S])
    K = np.floor(2*S*S/3)
    for k in range(Nr):
        leyend = 'Estimating Airlight: ' + str(int(100*(k+1)/Nr)) + '%'
        #print(leyend)
        for l in range(Nc):
            f_mv = ( f_proc[ k:S+k, l:S+l ] )
            f_max = f_mv.max()
            f_min = f_mv.min()
            #u = np.mean(f_mv[:])
            u = (f_min + f_max) / 2.0
            #v = (1 + np.var(f_mv[:]))
            v = f_max - f_min
            A_test[k,l] = u / (1 + v)

    x0,y0 = np.where( A_test == A_test.max())
    A = np.zeros(3)
    A[0] = f_c[x0[0], y0[0],0]
    A[1] = f_c[x0[0], y0[0],1]
    A[2] = f_c[x0[0], y0[0],2]
    A_est = 0.2989*A[0] + 0.5870*A[1] + 0.1140*A[2]

    t_est = np.zeros([Nr,Nc])
    trans = np.zeros([Nr,Nc])

    #PAR = [15, 0.01, 1.0, 4.] # Parameters for maguey
    PAR = [19, 0.7, 10.0, 4.0] # Parameters for flores 
    #PAR = [19, 0.6, 0.6, 6.] # Parameters for fuente

    S = 19      # Defines the size of sliding-window SxS
    w = y      # Minimum allowed value of transmission
    j0 = PAR[2]     # Parameter for transmission estimation
    Kdiv = PAR[3]   # Parameter for caculation of K in adaptive neighborhoods

    y = np.zeros([Nr,Nc,Np])
    K = S**2 - int((S**2)/Kdiv)
 

    for k in range(Nr):
        leyend = 'Estimating Transmission: ' + str(int(100*(k+1)/Nr)) + '%'
        #print(leyend)
        for l in range(Nc):
            f_w = f_proc[ k:S+k, l:S+l, : ]
            f_v = KNB(f_w, K)
            Fmax = f_v.max()
            Fmin = f_v.min()
            range_fv = Fmax - Fmin
            fv_avg = (Fmin+Fmax)/2.0
            if range_fv < w:
                range_fv = w 
            alpha = range_fv/(j0*A_est)
            t_est[k,l] = (A_est - (alpha*fv_avg + (1-alpha)*Fmin)) / (A_est-alpha*fv_avg)
                    
            if t_est[k,l] > 1:
                t_est[k,l] = 1
            if t_est[k,l] < w:
                t_est[k,l] = w

    trans = guidedFilter(np.uint8(f_c),np.uint8(255*t_est),30,0.1) / 255.0

    y[:,:,0] = (f_c[:,:,0] - A[0]) / trans + A[0]
    y[:,:,1] = (f_c[:,:,1] - A[1]) / trans + A[1] 
    y[:,:,2] = (f_c[:,:,2] - A[2]) / trans + A[2]
    y[:,:,:] = abs(y[:,:,:])
    xs,ys,zs = np.where( y > 255 )
    y[xs,ys,zs] = 255.0
                        
    #plt.subplot(121), plt.imshow(f_c/255.), plt.title('Hazy Image')
    #plt.subplot(122), plt.imshow(y/255.), plt.title('Processed Image')
    return plt.imshow(y/255.)




    

if __name__ == "__main__":
    main()


# {"mode":"full","isActive":false}