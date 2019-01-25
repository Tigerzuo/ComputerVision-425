from PIL import Image
import numpy as np
import math
from scipy import signal

#425

def boxfilter(n):
	assert n%2!=0, "Box size cannot be even!"
	if(n%2 != 0):
		a = np.full((n,n),1/(n*n))
		print(a)
		return a


def gauss1d(sigma):
	size = 6*math.ceil(sigma)+1
	sum = 0
	a = np.full(size,1,dtype=float)
	print(a)
	for i in range(size):
		exp = np.exp(-1 * ((sigma*3)-i)**2 / (2*sigma**2))
		sum += exp
		a[i] = exp
	np.set_printoptions(precision=5)
	a /=sum
	print(a)
	return a

def gauss2d(sigma):
	f = gauss1d(sigma)
	print(f.shape)
	f = f[np.newaxis]
	#print(f)
	#print(f.T)
	result = signal.convolve2d(f,f.T)
	#print("result",result)
	#print("result",result.shape)
	return result

def gaussconvolve2d(array,sigma):
	filter = gauss2d(sigma)
	result = signal.convolve2d(array,filter,mode="same")
	#difference bettween convolve and correlate
	return result

#seprable convolution filter

def gaussdog_greyscale():
    im = Image.open('dog.jpg')
    # convert the image to a black and white "luminance" greyscale image
    im2 = im.convert('L')
    im2_array = np.asarray(im2)
    im2_array = im2_array.astype(float)
    print("array type",im2_array)
    print("array shape", im2_array.shape)
    im3_array = im2_array.copy()      #make copy to change img

    img = gaussconvolve2d(im3_array,3)
    im2_plt = Image.fromarray(img)    # back to plt
    im2.show()
    im2_plt.show()

def gauss_pair(img_path,highfreq=False):
    im = Image.open(img_path)
    im2 = im.convert('RGB')
    im2_array = np.asarray(im2)
    im2_array = im2_array.astype(float)
    #print("array type", im2_array)
    #print("color array shape",im2_array.shape)

    img_final = np.zeros((361,410,3), 'uint8')             #convert back to one 3d array
    img_final[:,:,0] = gaussconvolve2d(im2_array[:,:,0],5) #cutoff freq
    img_final[:,:,1] = gaussconvolve2d(im2_array[:,:,1],5) #cutoff freq
    img_final[:,:,2] = gaussconvolve2d(im2_array[:,:,2],5) #cutoff freq


    if(highfreq):
    #highfreq code
        img2_copy = im2_array.copy()

        #high freq for visulization
        """
        img2_copy[:,:,0] -= img_final[:,:,0] + 128
        img2_copy[:,:,1] -= img_final[:,:,1] + 128
        img2_copy[:,:,2] -= img_final[:,:,2] + 128
        """
        img2_copy[:,:,0] -= img_final[:,:,0]
        img2_copy[:,:,1] -= img_final[:,:,1]
        img2_copy[:,:,2] -= img_final[:,:,2]

        img2_copy = img2_copy.astype('uint8')
        img_plt = Image.fromarray(img2_copy)
        #img_plt.show()
        return img_plt
    
    else:
        img_final = img_final.astype('uint8')
        img_plt = Image.fromarray(img_final)
        #img_plt.show()
        return img_plt

def addimg_channel(img1,img2):
    im1 = img1.convert('RGB')
    im2 = img2.convert('RGB')
    im1_a = np.asarray(im1)
    im2_a = np.asarray(im2)
    im1_a = im1_a.astype(float)
    im2_a = im2_a.astype(float)
    im1_copy = im1_a.copy()

    #split up into 3 channel
    im1_copy[:,:,0] += im2_a[:,:,0]
    im1_copy[:,:,1] += im2_a[:,:,1]
    im1_copy[:,:,2] += im2_a[:,:,2]

    im1_copy = np.clip(im1_copy,0,255)
    #np.clip(im1_copy[:,:,1],0,255)
    #np.clip(im1_copy[:,:,2],0,255)

    

    im1_copy = im1_copy.astype('uint8')
    print("combined image",im1_copy)
    img_plt = Image.fromarray(im1_copy)
    img_plt.show()


if __name__ == "__main__":
    #boxfilter(5)
    #gauss1d(0.3)
    #gauss2d(0.5)
    #gauss2d(1)

    #print("here")
    lowfreq_img  = gauss_pair('hw1/0b_dog.bmp')
    highfreq_img = gauss_pair('hw1/0a_cat.bmp',True)
    addimg_channel(lowfreq_img,highfreq_img)

    #gaussdog_greyscale();
