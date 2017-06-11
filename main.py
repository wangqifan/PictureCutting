import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import metrics
import sklearn.cluster as skc

def loadData(filepath):
    f=open(filepath,'rb')
    data=[]
    img=image.open(f)
    m,n=img.size
    for i in range(m):
        for j in range(n):
             x,y,z = img.getpixel((i,j))
             data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return np.mat(data),m,n


imgData,row,col=loadData("wqf.jpg")
print(row,col)
#db=skc.DBSCAN(eps=1,min_samples=20).fit(imgData)
#label=db.labels_
label = KMeans(n_clusters=20).fit_predict(imgData)
label=label.reshape([row,col])
pic_new =image.new("L",(row,col))
for i in range(row):
    for j in range(col):
       print(label[i][j])
       pic_new.putpixel((i,j),int(256/(label[i][j]+2)))
pic_new.save("2.jpg","JPEG")
