import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from networkx import nx

n = 500
m =5

def atac(trust,inf,DF,vnach,ob):
    vnach=0
    vnach=v1

    '''for i in range(n):
        for j in range(n):
            if i==j:
                trust[i][j]=1
                inf[i][j]=0
                DF[i][j]=1
            if DF[i][j]==0:
                trust[i][j]=0
                inf[i][j]=0
'''
    trust_atac=[]
    inf_atac=[]
    sigma_trust=[]
    sigma_tr=0
    sigma_i=0
    sigma_inf=[]
    porog=0.9
    rasp=0
    nrasp=[]

    for f in range(100):
        rasp=0
        for i in range(n):
            for j in range(n):
                if vnach[j]>=porog and DF[i][j]==1:
                    trust_atac.append(trust[i][j])
                    inf_atac.append(inf[j][i])
            for k in range(len(trust_atac)):
                if k==0:
                    sigma_tr=trust_atac[k]
                else:
                    sigma_tr=sigma_tr+(1-sigma_tr)*trust_atac[k]
            sigma_trust.append(sigma_tr)
            sigma_tr=0
            for k in range(len(inf_atac)):
                if k==0:
                    sigma_i=inf_atac[k]
                else:
                    sigma_i=sigma_i+(1-sigma_i)*inf_atac[k]  
            sigma_inf.append(sigma_i)
            sigma_i=0
            trust_atac.clear()
            inf_atac.clear()
   
        i=0
        for i in range(n):
            vnach[i]=vnach[i]+(1-vnach[i])*(sigma_inf[i]+(1-sigma_inf[i])*sigma_trust[i])*ob[i]
        sigma_inf=[]
        sigma_trust=[]
        for k in range(n):
            if vnach[k]>=porog:
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

def atac_za(trust,inf,DF,vnach,ob,za):
    vnach=0
    vnach=v1
    trust=[]
    trust=trust1
    inf=[]
    inf=inf1
    for i in range(n):
        for j in range(n):
            if i==j:
                trust[i][j]=1
                inf[i][j]=0
                DF[i][j]=1
            if DF[i][j]==0:
                trust[i][j]=0
                inf[i][j]=0

    trust_atac=[]
    inf_atac=[]
    sigma_trust=[]
    sigma_tr=0
    sigma_i=0
    sigma_inf=[]
    porog=0.9
    kolvo=0
    atac=0
    rasp=0
    pr=0
    z=5
    nrasp=[]

    for f in range(150):
        rasp=0
        for i in range(n):
            for j in range(n):
                if vnach[j]>=porog and DF[i][j]==1:
                    trust_atac.append(trust[i][j])
                    inf_atac.append(inf[j][i])
            for k in range(len(trust_atac)):
                if k==0:
                    sigma_tr=trust_atac[k]
                else:
                    sigma_tr=sigma_tr+(1-sigma_tr)*trust_atac[k]
            sigma_trust.append(sigma_tr)
            sigma_tr=0
            for k in range(len(inf_atac)):
                if k==0:
                    sigma_i=inf_atac[k]
                else:
                    sigma_i=sigma_i+(1-sigma_i)*inf_atac[k]  
            sigma_inf.append(sigma_i)
            sigma_i=0
            trust_atac.clear()
            inf_atac.clear()
   
        i=0
        for i in range(n):
            vnach[i]=vnach[i]+(1-vnach[i])*(sigma_inf[i]+(1-sigma_inf[i])*sigma_trust[i])*ob[i]
        sigma_inf=[]
        sigma_trust=[]
        for k in range(n):
            if vnach[k]>=porog:
                rasp=rasp+1
        nrasp.append(rasp)

            #наблюдатель просматривает по 5 агентов и выдает защиту
        for i in range(n):
            if vnach[i]>=porog:
                kolvo=kolvo+1
        if kolvo==0.5*n:
            atac=1
            
        while pr<z:
            if vnach[pr]>=porog and atac==0:
                za[pr]=za[pr]+(1-za[pr])*0.14
                for j in range(n):
                    if DF[pr][j]==1: 
                        za[j]=za[j]+(1-za[j])*0.1
            if vnach[pr]>=porog and atac==1:
                za[pr]=za[pr]+(1-za[pr])*0.4
                for j in range(n):
                    if DF[pr][j]==1: 
                        za[j]=za[j]+(1-za[j])*0.4
            pr=pr+1
        else:
            if z<n:
                z=z+5
            else:
                pr=0
                z=5
         #определение уровня защиты и изменение начального мнения
        for s in range(n):
            if za[s]>=0.38 and za[s]<0.86:
                vnach[s]=vnach[s]-0.1
            if za[s]>=0.86:
                vnach[s]=vnach[s]-0.15

    f=f+1       
    plt.grid(True)  
    plt.title("Распространение информации")
    plt.ylabel("Количество агентов распространяющих информацию")
    plt.xlabel("Время")
    plt.axis([0, f, 0, n+1])  
    plt.plot(nrasp)
    return nrasp

Ggen=nx.watts_strogatz_graph(n,m,0.15)
DFgen= pd.DataFrame(np.zeros([len(Ggen.nodes()),len(Ggen.nodes())]),index=Ggen.nodes(),columns=Ggen.nodes())
for col_label,row_label in Ggen.edges():
    DFgen.loc[col_label,row_label] = 1
    DFgen.loc[row_label,col_label] = 1
    DFgen=DFgen.astype(int)
#первое-среднее,второе-мат ожидание
trustgen=np.round(np.random.normal(0.4, 0.1, size=(n, n)), 3)
infgen = np.round(np.random.normal(0.4, 0.1, size=(n, n)), 3)
vgenir = np.round(np.random.normal(0.4, 0.1, size=n), 3)
zagen = np.round(np.random.normal(0.01, 0, size=n), 3)
obgen=np.round(np.random.normal(0.6, 0.1, size=n), 3)

index_trust = np.argmax(np.sum(trustgen,axis=0))
index_inf = np.argmax(np.sum(infgen,axis=1))
index_gen=np.random.randint(0,n)


G=Ggen
DF1=DFgen
v1=[]
v1.clear()
ob1=obgen
inf1=infgen
trust1=trustgen
for i in range(len(vgenir)):
    v1.append(vgenir[i])
v1[index_gen]=0.9


ver=[]
ol=2
for y in range(7):
    G=nx.watts_strogatz_graph(n,ol,0.05)
    DF1= pd.DataFrame(np.zeros([len(G.nodes()),len(G.nodes())]),index=G.nodes(),columns=G.nodes())
    for col_label,row_label in G.edges():
        DF1.loc[col_label,row_label] = 1
        DF1.loc[row_label,col_label] = 1
        DF1=DF1.astype(int)
    ver.append(atac(trust1,inf1,DF1,v1,ob1))
    np.savetxt('SosediVer.csv', ver, delimiter = ';', fmt='%d')
    trust1=trustgen
    inf1=infgen
    ob1=obgen
    v1.clear()
    for i in range(len(vgenir)):
        v1.append(vgenir[i])
    v1[index_gen]=0.9
    ol=ol+1
plt.title("Влияние соседей на распространение информации")
plt.figure(figsize=(10,10))
plt.show()

ver=[]
p=0.05
for y in range(7):
    G=nx.watts_strogatz_graph(n,m,p)
    DF1= pd.DataFrame(np.zeros([len(G.nodes()),len(G.nodes())]),index=G.nodes(),columns=G.nodes())
    for col_label,row_label in G.edges():
        DF1.loc[col_label,row_label] = 1
        DF1.loc[row_label,col_label] = 1
        DF1=DF1.astype(int)
    ver.append(atac(trust1,inf1,DF1,v1,ob1))
    np.savetxt('Veroyat.csv', ver, delimiter = ';', fmt='%d')
    trust1=trustgen
    inf1=infgen
    v1.clear()
    ob1=obgen
    for i in range(len(vgenir)):
        v1.append(vgenir[i])
    v1[index_gen]=0.9
    p=p+0.05
plt.title("Влияние вероятности на распространение информации")
plt.figure(figsize=(10,10))
plt.show()

ver=[]
G=Ggen
DF1=DFgen
u=0.1
for y in range(6):
    trust1=np.round(np.random.normal(u, 0.05, size=(n, n)), 3)
    ver.append(atac(trust1,inf1,DF1,v1,ob1))
    np.savetxt('Doverie.csv', ver, delimiter = ';', fmt='%d')
    v1.clear()
    G=Ggen
    DF1=DFgen
    ob1=obgen
    inf1=infgen
    for i in range(len(vgenir)):
        v1.append(vgenir[i])
    v1[index_gen]=0.9
    u=u+0.1
plt.title("Влияние степени доверенности сети на распространение информации")
plt.figure(figsize=(10,10))
plt.show()

ver=[]
trust1=trustgen
w=0.1
for y in range(6):
    inf1=np.round(np.random.normal(w, 0.05, size=(n, n)), 3)
    ver.append(atac(trust1,inf1,DF1,v1,ob1))
    np.savetxt('Doverie.csv', ver, delimiter = ';', fmt='%d')
    trust1=trustgen
    G=Ggen
    DF1=DFgen
    v1.clear()
    ob1=obgen
    for i in range(len(vgenir)):
        v1.append(vgenir[i])
    v1[index_gen]=0.9
    w=w+0.1
plt.title("Влияние степени влиятельности сети на распространение информации")
plt.figure(figsize=(10,10))
plt.show()

ver=[]
v1.clear()
for i in range(len(vgenir)):
    v1.append(vgenir[i])
trust1=trustgen
inf1=infgen
DF1=DFgen
ob1=obgen
v1[index_trust]=0.9
ver.append(atac(trust1,inf1,DF1,v1,ob1))
np.savetxt('Mestopol.csv', ver, delimiter = ';', fmt='%d')
plt.title("Влияние местоположения агента на распространение информации (самый доверенный)")
plt.figure(figsize=(10,10))
plt.show()

v1.clear()
for i in range(len(vgenir)):
    v1.append(vgenir[i])
trust1=trustgen
inf1=infgen
DF1=DFgen
ob1=obgen
v1[index_inf]=0.9
ver.append(atac(trust1,inf1,DF1,v1,ob1))
np.savetxt('Mestopol.csv', ver, delimiter = ';', fmt='%d')
plt.title("Влияние местоположения агента на распространение информации (самый влиятельный)")
plt.figure(figsize=(10,10))
plt.show()


v1.clear()
for i in range(len(vgenir)):
    v1.append(vgenir[i])
trust1=trustgen
inf1=infgen
DF1=DFgen
ob1=obgen
v1[index_gen]=0.9
ver.append(atac(trust1,inf1,DF1,v1,ob1))
np.savetxt('Mestopol.csv', ver, delimiter = ';', fmt='%d')
plt.title("Влияние местоположения агента на распространение информации (случайный)")
plt.figure(figsize=(10,10))
plt.show()

ver=[]
v1.clear()
for i in range(len(vgenir)):
    v1.append(vgenir[i])
trust1=trustgen
inf1=infgen
DF1=DFgen
ob1=obgen
v1[index_gen]=0.9
za1=zagen
ver.append(atac_za(trust1,inf1,DF1,v1,ob1,za1))
np.savetxt('ZA.csv', ver, delimiter = ';', fmt='%d')
plt.title("Два процесса в защитой одновременно")
plt.figure(figsize=(10,10))
plt.show()

