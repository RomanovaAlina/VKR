import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from networkx import nx

n = 250
m =5
    
def func(trust,inf,DF,vnach):
    cowrow=trust.shape
    cow=cowrow[0]
    row=cowrow[1]
    vnach=0
    vnach=v
    for i in range(cow):
        for j in range(row):
            if i==j:
                trust[i][j]=1
                inf[i][j]=0
                DF[i][j]=1
            if DF[i][j]==0:
                trust[i][j]=0
                inf[i][j]=0
    a=0
    b=0
    i=0
    sumtrust=[]
    suminf=[]
    rasp=0
    nrasp=[]

    for f in range(50):
        i=0
        j=0
        k=0
        rasp=0
        sumtrust.clear()
        suminf.clear()

        for i in range(cow):
            for j in range(row):
                if vnach[j]==4 and DF[i][j]==1:
                    a=trust[i][j]+a
                    b=inf[j][i]+b
            sumtrust.append(a)
            suminf.append(b)
            a=0
            b=0
    
        for k in range(cow):
            if (vnach[k]==0 or vnach[k]==1 or vnach[k]==2 or vnach[k]==3) and suminf[k]>=1:
                if sumtrust[k]>=1:
                    vnach[k]=vnach[k]+1
                else:
                   vnach[k]=vnach[k]+1
            if (vnach[k]==0 or vnach[k]==1 or vnach[k]==2 or vnach[k]==3) and sumtrust[k]>=1 and suminf[k]==0:
                vnach[k]=vnach[k]+1
        k=0        
        for k in range(cow):
            if vnach[k]>=4:
                rasp=rasp+1
        nrasp.append(rasp)

    f=f+1       
    plt.grid(True)  
    plt.title("Распространение информации")
    plt.ylabel("Количество агентов распространяющих информацию")
    plt.xlabel("Время")
    plt.axis([0, f, 0, n+1])  
    plt.plot(nrasp)
    return nrasp

def atac_za(trust,inf,DF,vnach,za):
    
    cowrow=trust.shape
    cow=cowrow[0]
    row=cowrow[1]
    vnach=0
    vnach=v
    for i in range(cow):
        for j in range(row):
            if i==j:
                trust[i][j]=1
                inf[i][j]=0
                DF[i][j]=1
            if DF[i][j]==0:
                trust[i][j]=0
                inf[i][j]=0
    a=0
    b=0
    i=0
    pr=0
    z=5
    sumtrust=[]
    suminf=[]
    predup=(np.random.rand(n,) > 1).astype(float)
    rasp=0
    nrasp=[]
        
    for f in range(120):
        i=0
        j=0
        k=0
        rasp=0
        sumtrust.clear()
        suminf.clear()
        
        for i in range(cow):
            for j in range(row):
                if vnach[j]==4 and DF[i][j]==1:
                    a=trust[i][j]+a
                    b=inf[j][i]+b
            sumtrust.append(a)
            suminf.append(b)
            a=0
            b=0
            
        if f<=0:          
            for k in range(cow):
                if (vnach[k]==0 or vnach[k]==1 or vnach[k]==2 or vnach[k]==3) and suminf[k]>=1:
                    if sumtrust[k]>=1:
                        vnach[k]=vnach[k]+1
                    else:
                        vnach[k]=vnach[k]+1
                if (vnach[k]==0 or vnach[k]==1 or vnach[k]==2 or vnach[k]==3) and sumtrust[k]>=1 and suminf[k]==0:
                    vnach[k]=vnach[k]+1     
                    
            k=0 
            for e in range(cow):
                if vnach[e]>=4:
                    rasp=rasp+1
            nrasp.append(rasp)

        else: 
            #наблюдатель просматривает по 5 агентов и выдает предупреждения
            while pr<z:
                if vnach[pr]>=4:
                    predup[pr]=predup[pr]+1
                for j in range(row):
                    if DF[pr][j]==1: 
                        predup[j]=predup[j]+0.5
                pr=pr+1
            else:
                if z<cow:
                    z=z+5
                else:
                    pr=0
                    z=5
            #если есть 5 предупреждений, то появляется защита
            for s in range(cow):
                if predup[s]==0:
                    za[s]=0
                if predup[s]==1 or predup[s]==2:
                    za[s]=1
                if predup[s]==3 or predup[s]==4:
                    za[s]=2  
                if predup[s]>=5:
                    za[s]=3  
            for i in range(cow):  
                if za[i]==0 or za[i]==1:
                    vnach[i]=vnach[i]
                if za[i]==2:
                    vnach[i]=vnach[i]-1    
                if za[i]==3:
                    vnach[i]=vnach[i]-2                     
                
            #если больше 5 предупреждений, то блокировка???
            
            for k in range(cow):
                if (vnach[k]==0 or vnach[k]==1 or vnach[k]==2 or vnach[k]==3) and suminf[k]>=1:
                    if sumtrust[k]>=1:
                        vnach[k]=vnach[k]+1
                    else:
                        vnach[k]=vnach[k]+1
                if (vnach[k]==0 or vnach[k]==1 or vnach[k]==2 or vnach[k]==3) and sumtrust[k]>=1 and suminf[k]==0:
                    vnach[k]=vnach[k]+1

                                
            for e in range(cow):
                if vnach[e]>=4:
                    rasp=rasp+1
            nrasp.append(rasp)
    f=f+1       
    plt.grid(True)  
    plt.title("Распространение информации")
    plt.ylabel("Количество агентов распространяющих информацию")
    plt.xlabel("Время")
    plt.axis([0, f, 0, n+1])  
    plt.plot(nrasp)
    return nrasp

#генерация всех характеристик нейтральногог вида
Ggen=nx.watts_strogatz_graph(n,m,0.1)
#nx.write_gexf(G, "test2.gexf") 
DFgen= pd.DataFrame(np.zeros([len(Ggen.nodes()),len(Ggen.nodes())]),index=Ggen.nodes(),columns=Ggen.nodes())
for col_label,row_label in Ggen.edges():
    DFgen.loc[col_label,row_label] = 1
    DFgen.loc[row_label,col_label] = 1
    DFgen=DFgen.astype(int)
trustgen = (np.random.rand(n,n) > 0.5).astype(int)
infgen = (np.random.rand(n,n) > 0.5).astype(int)
vgen = np.random.random_sample((n,))
v2 = np.random.random_sample((n,))
for i in range(n):
    if v2[i]>=0.3:
        vgen[i]=2 #абсолютное большинство нейтральное
    if v2[i]<=0.075 and v2[i]>0:
        vgen[i]=0 #сильно негативное
    if v2[i]<=0.15 and v2[i]>0.075:
        vgen[i]=1 #негативное
    if v2[i]<=0.225 and v2[i]>0.15:
        vgen[i]=2 #нейтральное      
    if v2[i]<0.3 and v2[i]>0.225:
        vgen[i]=3 # положительно 
index_trust = np.argmax(np.sum(trustgen,axis=0))
index_inf = np.argmax(np.sum(infgen,axis=1))
index_gen=np.random.randint(0,n)
zagen =(np.random.rand(n,) > 1).astype(int)
zaindex_gen=np.random.randint(0,n)

za1=zagen


v=vgen.astype(int)
v[index_gen]=4
trust1=trustgen.astype(int)
inf1=infgen.astype(int)
G=Ggen
DF1=DFgen

'''
csv=[]
ol=4
for y in range(9):
    G=nx.watts_strogatz_graph(n,ol,0.15)
    DF1= pd.DataFrame(np.zeros([len(G.nodes()),len(G.nodes())]),index=G.nodes(),columns=G.nodes())
    for col_label,row_label in G.edges():
        DF1.loc[col_label,row_label] = 1
        DF1.loc[row_label,col_label] = 1
        DF1=DF1.astype(int)
    #func(trust1,inf1,DF1,v)
    csv.append(func(trust1,inf1,DF1,v))
    np.savetxt('Sosedi.csv', csv, delimiter = ';', fmt='%d')
    v=vgen.astype(int)
    v[index_gen]=4
    trust1=trustgen.astype(int)
    inf1=infgen.astype(int)
    G=Ggen
    DF1=DFgen
    ol=ol+1
plt.title("Влияние соседей на распространение информации")
plt.figure(figsize=(10,10))
plt.show()


csv=[]
p=0.05
for y in range(9):
    G=nx.watts_strogatz_graph(n,m,p)
    DF1= pd.DataFrame(np.zeros([len(G.nodes()),len(G.nodes())]),index=G.nodes(),columns=G.nodes())
    for col_label,row_label in G.edges():
        DF1.loc[col_label,row_label] = 1
        DF1.loc[row_label,col_label] = 1
        DF1=DF1.astype(int)
    #func(trust1,inf1,DF1,v)
    csv.append(func(trust1,inf1,DF1,v))
    np.savetxt('Veroyat.csv', csv, delimiter = ';', fmt='%d')
    v=vgen.astype(int)
    v[index_gen]=4
    trust1=trustgen.astype(int)
    inf1=infgen.astype(int)
    G=Ggen
    DF1=DFgen
    p=p+0.1
plt.title("Влияние вероятности на распространение информации")
plt.figure(figsize=(10,10))
plt.show()

csv=[]
u=0.1
for y in range(9):
    trust1 = (np.random.rand(n,n) > u).astype(int)
    csv.append(func(trust1,inf1,DF1,v))
    np.savetxt('Doverie.csv', csv, delimiter = ';', fmt='%d')
    v=vgen.astype(int)
    v[index_gen]=4
    trust1=trustgen.astype(int)
    inf1=infgen.astype(int)
    G=Ggen
    DF1=DFgen
    u=u+0.1
plt.title("Влияние степени доверенности сети на распространение информации")
plt.figure(figsize=(10,10))
plt.show()

csv=[]
w=0.1
for y in range(9):
    inf1 = (np.random.rand(n,n) > w).astype(int)  
    csv.append(func(trust1,inf1,DF1,v))
    np.savetxt('Infl.csv', csv, delimiter = ';', fmt='%d')
    v=vgen.astype(int)
    v[index_gen]=4
    trust1=trustgen.astype(int)
    inf1=infgen.astype(int)
    G=Ggen
    DF1=DFgen
    w=w+0.1
plt.title("Влияние степени влиятельности сети на распространение информации")
plt.figure(figsize=(10,10))
plt.show()

csv=[]
w=0
for y in range(4):
    v1 = np.random.random_sample((n,))
    v2 = np.random.random_sample((n,))
    for i in range(n):
        if v2[i]>=0.3:
            v1[i]=w #абсолютное большинство нейтральное
        if v2[i]<=0.075 and v2[i]>0:
            v1[i]=0 #сильно негативное
        if v2[i]<=0.15 and v2[i]>0.075:
            v1[i]=1 #негативное
        if v2[i]<=0.225 and v2[i]>0.15:
            v1[i]=2 #нейтральное      
        if v2[i]<0.3 and v2[i]>0.225:
            v1[i]=3 # положительно 
    v=v1.astype(int)
    v[index_gen]=4
    csv.append(func(trust1,inf1,DF1,v))
    np.savetxt('Vnach.csv', csv, delimiter = ';', fmt='%d')
    v=vgen.astype(int)
    trust1=trustgen.astype(int)
    inf1=infgen.astype(int)
    G=Ggen
    DF1=DFgen
    w=w+1
plt.title("Влияние нач состояния на распространение информации")
plt.figure(figsize=(10,10))
plt.show()


csv=[]
v=vgen.astype(int)
trust1=trustgen.astype(int)
inf1=infgen.astype(int)
G=Ggen
DF1=DFgen
v[index_trust]=4
csv.append(func(trust1,inf1,DF1,v))
np.savetxt('Mestopol.csv', csv, delimiter = ';', fmt='%d')
plt.title("Влияние местоположения агента на распространение информации (самый доверенный)")
plt.figure(figsize=(10,10))
plt.show()

v=vgen.astype(int)
trust1=trustgen.astype(int)
inf1=infgen.astype(int)
G=Ggen
DF1=DFgen
v[index_inf]=4
csv.append(func(trust1,inf1,DF1,v))
np.savetxt('Mestopol.csv', csv, delimiter = ';', fmt='%d')
plt.title("Влияние местоположения агента на распространение информации (самый влиятельный)")
plt.figure(figsize=(10,10))
plt.show()

v=vgen.astype(int)
trust1=trustgen.astype(int)
inf1=infgen.astype(int)
G=Ggen
DF1=DFgen
v[index_gen]=4
csv.append(func(trust1,inf1,DF1,v))
np.savetxt('Mestopol.csv', csv, delimiter = ';', fmt='%d')
plt.title("Влияние местоположения агента на распространение информации (случайный)")
plt.figure(figsize=(10,10))
plt.show()

csv=[]
w=1
for y in range(8):
    v=vgen.astype(int)
    v[index_gen]=4
    trust1=trustgen.astype(int)
    inf1=infgen.astype(int)
    G=Ggen
    DF1=DFgen
    za1=zagen.astype(int)
    za1 = (np.random.rand(n,) > w).astype(int)  
    csv.append(atac_za(trust1,inf1,DF1,v,za1))
    np.savetxt('za.csv', csv, delimiter = ';', fmt='%d')
    w=w-0.1
plt.title("Влияние заищенного количества агентов на распространение информации")
plt.figure(figsize=(10,10))
plt.show()
'''
csv=[]
v=vgen.astype(int)
v[index_gen]=4
trust1=trustgen.astype(int)
inf1=infgen.astype(int)
DF1=DFgen
za1=zagen.astype(int)  
csv.append(atac_za(trust1,inf1,DF1,v,za1))
np.savetxt('ZAMestopol.csv', csv, delimiter = ';', fmt='%d')
plt.title("Влияние местоположения агента с защитой на распространение информации (без защиты)")
plt.figure(figsize=(10,10))
plt.show()


print("Результаты сохранены")   
