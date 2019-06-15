import pandas as panda
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import matplotlib.cm as cm
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from PIL import ImageTk, Image
import os
current_path = os.path.dirname(os.path.abspath(__file__))

import tkinter as tk
from tkinter import ttk

LARGE_FONT= ("Verdana", 12)
SMALL_FONT= ("Verdana", 10)


root = tk.Tk()

root.wm_title("K-means")

#icon_image_path = r'C:\Users\stanshi\Desktop\Sp.sp_stan_icon.ico'
#root.iconbitmap(icon_image_path)
canvas1 = tk.Canvas(root, width = 250, height = 250)
canvas1.pack()

#add Background image
bg_image_path = r''+current_path+'\Sp.stan_logo_180px.png'#r'C:\Users\stanshi\Desktop\Sp.stan_logo_180px.png'
bg_image = ImageTk.PhotoImage(Image.open(bg_image_path)) # PIL module
#bg_img = tk.Label(root, image = bg_image)
#bg_img.pack(side = "top", fill = "both", expand = "yes")
canvas1.create_image(125, 40, image = bg_image)

label_cluster = tk.Label(text="Enter no. of cluster(k): ", font=LARGE_FONT)
canvas1.create_window(125, 100, window=label_cluster) 

label_author = tk.Label(text="Author: Fairuz Safwan", font=SMALL_FONT)
canvas1.create_window(125, 240, window=label_author) 

button1 = tk.Button (root, text='Exit', command=root.destroy)
canvas1.create_window(180, 170, window=button1) 

#text box for number of cluster
textbox_k = tk.Entry (root)
canvas1.create_window(120, 130, window=textbox_k) 

#initialize number of cluster
k = 0

#To compute euclidean distance between points, axis=1: to compute distance for all along x-axis, axis=none: compute vector norm for 1-D
def eudistance(p1,p2, axisParam):
    return np.linalg.norm((p1-p2), axis=axisParam)

def updateCentroids(cx, cy, p):
    for a in range(len(cx)):
        cx[a] = cx[a]/p[a]
        cy[a] = cy[a]/p[a]
    return cx, cy

def getK():
    #fig = Figure(figsize=(6,6))
    #a = fig.add_subplot(111)

    #Start timer
    start = time.time()
	
	#get input from textbox
    k = int(textbox_k.get()) 
    #k = int(sys.argv[1]) # number of cluster from command line
	
    centroidsX = []
    centroidsY = []

    #read data from file
    data = panda.read_csv("xclara.csv") 
    print(data.shape)
    x = data["V1"].values
    y = data["V2"].values
    arrayXY = np.array(data)



    #initialize random centroids
    centroidsX = np.array(np.random.randint(0, np.amax(x), size=k))
    centroidsY = np.array(np.random.randint(0, np.amax(y), size=k))

    #Merge centroids coordinate x,y into list
    cXY = np.array(list(zip(centroidsX, centroidsY)))

    print("Initial centroids: ")
    print(cXY)

    centX_new = []
    centY_new = []
    centroids_new = []
    iterations = 0
    centroids_error = 1 #to proceed into loop

    while(centroids_error != 0):
        print("Iteration no.: " + str(iterations+1))
        pointToCluster = []
        averageX = np.zeros(k)
        averageY = np.zeros(k)
        numPoints = np.zeros(k)
        for i in range(len(x)):
            dist = eudistance(arrayXY[i], cXY, 1)
            pointToCluster.append((np.argmin(dist), arrayXY[i])) # gets minimum/shortest index and value among clusters and append to array
            averageX[np.argmin(dist)] += arrayXY[i][0]
            averageY[np.argmin(dist)] += arrayXY[i][1]
            numPoints[np.argmin(dist)] += 1


        centX_new, centY_new = updateCentroids(averageX, averageY, numPoints)
        centroids_new = np.array(list(zip(centX_new, centY_new)))

        print("previous Centroids: ")
        print(cXY)
        print("Updated Centroids: ")
        print(centroids_new)

        centroids_error = eudistance(cXY, centroids_new, None) #to determine if centroids need update otherwise stop iteration
        print("Centroids error: ")
        print(centroids_error)
        cXY = centroids_new
        iterations += 1
        print()



    #Colourize each points
    #time consuming - needs optimization
    #other colour options: 'teal', 'm','papayawhip', 'y', 'k', 
    colours = ['b', 'g', 'r', 'c', 'aliceblue', 'aqua', 'forestgreen', 'deeppink', 'blanchedalmond', 'burlywood', 'darkgoldenrod'] 
    for o in range(k):
        for t in range(len(pointToCluster)):
            if pointToCluster[t][0] == o:
                plt.scatter(pointToCluster[t][1][0], pointToCluster[t][1][1], c=colours[o], s=10)


    for i in range(k):
        c_old_legend = plt.scatter(centroidsX[i], centroidsY[i], c="black", s=50)
        plt.text(centroidsX[i]+0.5, centroidsY[i]+0.9, "C" + str(i+1), fontsize=9, color="k")
        c_new_legend = plt.scatter(centX_new[i], centY_new[i], marker="*", c="yellow", s=50)
        plt.text(centX_new[i]+0.5, centY_new[i]+0.9, "C " + str(i+1), fontsize=9, color="k")
    plt.legend((c_old_legend, c_new_legend),
               ('Initial Centroid Coordinate', 'Latest Centroid Coordinate'),
               scatterpoints=1,
               loc='lower left',
               ncol=3,
               fontsize=8)
    plt.savefig("kmeans.png")

    #Stop timer
    end = time.time()

    #To embed figure into TKinter - works! but have to use fig instead of plt
    #canvas1 = FigureCanvasTkAgg(fig)
    #canvas1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
    #canvas1.draw()
    
    #Write to kmeans.txt file
    f = open("kmeans.txt","w+")
    f.write("===============================\n")
    f.write("Time taken (s): ")
    f.write(str(end - start))
    f.write("\n")
    f.write("Iterations: " + str(iterations))
    f.write("\n")

    #output summary to command line
    print("===============================")
    print("Time taken (s): ")
    print(end - start)
    print("Iterations: " + str(iterations))
    print("Latest Centroids: ")


    f.write("Centroids: \n")
    for i in range(len(centroids_new)):
        f.write("Centroid " + str(i+1) + " = ")
        print("Centroid " + str(i+1) + " = " + str(centroids_new[i][0]) + ", " + str(centroids_new[i][1]))
        f.write(str(centroids_new[i][0]))
        f.write(", ")
        f.write(str(centroids_new[i][1]))
        f.write("\n")
    print("Centroids ID - Points: --Please refer to the save file named kmeans.txt--")
    print("===============================")


    f.write("Centroid ID - Points: \n")
    for i in range(len(pointToCluster)):
        f.write(str(pointToCluster[i][0]))
        f.write(" - ")
        f.write(str(pointToCluster[i][1][0]))
        f.write(", ")
        f.write(str(pointToCluster[i][1][1]))
        f.write("\n")
    f.write("===============================")
    f.close()

    plt.show()
	
button2 = tk.Button (root, text='Compute K-means', command=getK)
canvas1.create_window(97, 170, window=button2) 

root.mainloop()