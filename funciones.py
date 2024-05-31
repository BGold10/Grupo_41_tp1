import numpy as np
import networkx as nx
from scipy.linalg import solve_triangular 



matriz1 = np.array([[0,1,1],[1,0,1],[1,1,0]])

def leer_archivo(input_file_path):

    f = open(input_file_path, 'r')
    n = int(f.readline())
    m = int(f.readline())
    W = np.zeros(shape=(n,n))
    for _ in range(m):
            line = f.readline()
            i = int(line.split()[0]) - 1
            j = int(line.split()[1]) - 1
            W[j,i] = 1.0
    f.close()
    
    return W

def dibujarGrafo(W, print_ejes=True):
    
    options = {
    'node_color': 'yellow',
    'node_size': 200,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 10,
    'with_labels' : True}
    
    N = W.shape[0]
    G = nx.DiGraph(W.T)
    
    #renombro nodos de 1 a N
    G = nx.relabel_nodes(G, {i:i+1 for i in range(N)})
    if print_ejes:
        print('Ejes: ', [e for e in G.edges])
    
    nx.draw(G, pos=nx.spring_layout(G), **options)
    plt.show()

def calcularRanking(M, p):
    npages = M.shape[0]
    rnk = np.arange(0, npages) 
    scr = np.zeros(npages) 
    D = ArmemosD(M)
    Identidad = np.eye(M.shape[0],M.shape[1])
    A_rara = Identidad - p * (M @ D)
    e = np.ones((M.shape[0], 1))   
    scr = LU_solve(A_rara, e)
    scr_normalizado = scr/sum(scr)
    rnk = ranking(scr)
    return rnk, scr_normalizado

def obtenerMaximoRankingScore(M, p):
    rnk, scr = calcularRanking(M, p)
    output = np.max(scr)
    return output

def ranking(scr): # ejemplo de lo que busca hacer la funcion ranking: Dado unos puntajes [0.2,0.5,0.3], el ranking seria [3,1,2]
    scrLista = scr.flatten().tolist()
    rnk = scrLista.copy()
    scrOrdenado = []
    scr2 = []
    for count, ele in enumerate(scrLista): #count es el numero del nodo, ele es el puntaje de cada nodo
        scrOrdenado.append((ele,count)) 
    scrOrdenado.sort(reverse = True) #ordeno la lista de los puntajes segun ele. Count queda desordenado
    n = 0
    for i in range(len(scrOrdenado)):
        scr2.append(scrOrdenado[i][1]) #en scr2 voy agregando el numero de nodo correspondiente a cada puntaje
    for i in scr2: #i va a recorrer los numeros de nodos del que tiene mayor puntaje al de menor puntaje
        n += 1 
        rnk[i] = n #en la posicion i de rnk le asigno la posicion del nodo i
    return rnk




def ArmemosD(M):
    D = np.zeros((M.shape[0],M.shape[1]))
    for j in range(M.shape[0]):
        cj = np.sum(M[j, :])
        if (cj!=0):
            D[j,j] = 1/cj
        else:
            D[j,j] = 0
    return D



def descomposicion_lu(A):
    n = A.shape[0] # n = cantidad de filas, coincide con el orden de la matriz
    L = np.eye(n)  # Creamos la matriz del mismo tamaño de A pero la matriz identidad
    U = A.astype(float)          # La matriz U comienza siendo una copia de A

    for j in range(n):                               # Recorremos las columnas
        if U[j, j] != 0:                             # Si el pivote es distinto de 0
            for i in range(j+1, n):                  # Convertimos en ceros los elementos bajo el pivote
                L[i, j] = U[i, j]/U[j, j]            # Escribimos el coeficiente (elemento / pivote) que se utiliza en Gauss en la matriz L       
                U[i, :] = U[i, :] - L[i, j]*U[j, :]  # Convertimos ese elemento en 0 en la matriz U
        else:
            print('Error: Hubo un 0 en la diagonal (El pivote es 0 y no existe factorizacion LU)')
            return np.eye(n), A                      # Si el pivote es 0, se devuelve la identidad y la matriz A

    return L, U


def LU_solve(A, b):
    L, U = descomposicion_lu(A) # Reutilizamos la funcion del punto anterior
    y = solve_triangular(L, b, lower=True) # El vector 'y' tiene la solucion del sistema Ly=b . lower=True significa que la matriz es triangular inferior.
    x = solve_triangular(U, y, lower=False) # El vector 'x' tiene la solucion que buscamos (de Ax=b). lower=False significa que la matriz es triangular inferior.
    return x


"""
Analisis cuantitativo
"""
import matplotlib
import matplotlib.pyplot as plt
import time
import random
from pylab import *   

def cant_links(M):
    links = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            links += M[i][j] 
    return links
def cant_nodos(M):
    return M.shape[0]

def graficoCuantitativoLinks(p):
    archivos = ["tests/test_dosestrellas.txt", "tests/mathworld_grafo.txt", "tests/test_15_segundos.txt", "tests/test_30_segundos.txt", "tests/test_aleatorio.txt", "tests/instagram_famosos_grafo.txt"]
    links = []
    tiempo = []
    for i in range(1,10):
        a = matricesAleatorias(i*50)
        links.append(cant_links(a))
        tiempo.append(tiempoDeCalculo(a, p))
    for i in range(6):
        W = leer_archivo(archivos[i])
        links.append(cant_links(W))
        tiempo.append(tiempoDeCalculo(W, p))
    x = np.array(links)
    y = np.array(tiempo)
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    plt.xlabel('Cantidad de links')
    plt.ylabel('Tiempo de Computos')
    plt.xscale("log")
    a = ("densidad segun los links con P= ", p)
    plt.title(a)
    
def analisisCuantitativo():
    for i in range(3,10,5):
        graficoCuantitativoLinks(i/10)
        
        graficoCuantitativoNodos(i/10)
        
        
def graficoCuantitativoNodos(p):
    archivos = ["tests/test_dosestrellas.txt", "tests/mathworld_grafo.txt", "tests/test_15_segundos.txt", "tests/test_30_segundos.txt", "tests/test_aleatorio.txt", "tests/instagram_famosos_grafo.txt"]
    nodos = []
    tiempo = []
    for i in range(1,8):
        a = matricesAleatorias(i*200)
        nodos.append(cant_nodos(a))
        tiempo.append(tiempoDeCalculo(a, p))
    for i in range(6):
        W = leer_archivo(archivos[i])
        nodos.append(cant_nodos(W))    #rnk_diccionario = {indice: valor for indice, valor in enumerate(rnk_normalizado.tolist(), 1)}
        tiempo.append(tiempoDeCalculo(W,p))
    x = np.array(nodos)
    y = np.array(tiempo)
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    plt.xscale("log")
    plt.xlabel('Cantidad de nodos')
    plt.ylabel('Tiempos de Computo')
    a = ("densidad segun los nodos con P= ", p)
    plt.title(a)

def matricesAleatorias(n):
    a = []
    for j in range(n):    
        m = []
        for i in range(n):
            p = random(1)
            if p > 0.1:
                m.append(1)
            else:
                m.append(0)
        a.append(m)
    return np.array(a)
def elMayor(M,p):
    rnk, scr = calcularRanking(M, p)
    scrlista = scr.flatten().tolist()
    mejor = scrlista[0]
    for i in range(len(scrlista)):
        if scrlista[i] > mejor:
            mejor = scrlista[i]
    output = mejor
    scrlista.remove(output)
    return output, scrlista

def los3mejores(M,p):
    output = []
    rnk, scr = calcularRanking(M, p)
    scrlista = scr.flatten().tolist()
    for i in range(3):
        mejor = scrlista[0]
        for i in range(len(scrlista)):
            if scrlista[i] > mejor:
                mejor = scrlista[i]
        output.append(mejor)
        scrlista.remove(mejor)
    return output

def graficoCualitativo(M):
    y1 = []
    x1 = []
    y3 = []
    x3 = []
    y2 = []
    x2 = []
    for j in range(3):
        x = []
        y = []
        for i in range(1,10):
            y.append(los3mejores(M, i/10)[j])
            x.append(i/10)
        if j == 0:
            x1 = x
            y1 = y
        elif j == 1:
            x2 = x
            y2 = y
        else:
            x3 = x
            y3 = y
    
    plt.yscale("linear")
    plt.xlabel('P')
    plt.ylabel('Scores')
    plt.title("3 primeros puestos en funcion de p")
    p1, p2, p3 = plot(x1, y1, x2, y2, x3, y3)
        
def tiempoDeCalculo(M,p):
    inicio = time.time()
    rnk,scr = calcularRanking(M, p)
    fin = time.time()
    return fin-inicio
