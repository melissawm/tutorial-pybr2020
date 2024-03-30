---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from IPython.display import Image
Image("imagens/numpylogo.png", width=600)
```

<!-- #region tags=["chapter"] -->
# NumPy: a base da computação científica no Python
<!-- #endregion -->

```python
import numpy as np
```

## O que é a NumPy?

> A computação com arrays é a base para estatística e matemática computacionais, computação científica e suas várias aplicações em ciência e análise de dados, tais como visualização de dados, processamento de sinais digitais, processamento de imagens, bioinformática, aprendizagem de máquina, IA e muitas outras.
>
> A manipulação e a transformação de dados de grande escala dependem de computação eficiente de alta performance com arrays. A linguagem mais escolhida para análise de dados, aprendizagem de máquina e computação numérica produtiva é Python.
>
> Numerical Python (Python Numérico) ou NumPy é a biblioteca em Python padrão para o suporte à utilização de matrizes e arrays  multidimensionais de grande porte, e vem com uma vasta coleção de funções matemáticas de alto nível para operar nestas arrays.
>
> Desde o lançamento do NumPy em 2006, o Pandas apareceu em 2008, e nos últimos anos vimos uma sucessão de bibliotecas de computação com arrays aparecerem, ocupando e preenchendo o campo da computação com arrays. Muitas dessas bibliotecas mais recentes imitam recursos e capacidades parecidas com o NumPy e entregam algoritmos e recursos mais recentes voltados para aplicações de aprendizagem de máquina e inteligência artificial.
>
> A computação com arrays é baseada em estruturas de dados chamadas arrays. Arrays são usadas para organizar grandes quantidades de dados de forma que um conjunto de valores relacionados possa ser facilmente ordenado, obtido, matematicamente manipulado e transformado fácil e rapidamente.
>
> A computação com arrays é única pois envolve operar nos valores de um array de dados de uma vez. Isso significa que qualquer operação de array se aplica a todo um conjunto de valores de uma só vez. Esta abordagem vetorizada fornece velocidade e simplicidade por permitir que os programadores organizem o código e operem em agregados de dados, sem ter que usar laços com operações escalares individuais.

(Fonte: [https://numpy.org](https://numpy.org))

A biblioteca NumPy consiste em:

- objeto `ndarray` (array homogêneo n-dimensional)
- capacidade de **broadcasting**
- funções matemáticas padrão com capacidade de vetorização
- ferramentas para a integração de código C/C++ e Fortran
- álgebra linear, transformadas de Fourier, gerador de números aleatórios

<!-- #region tags=["section"] -->
### Por que não usar listas?
<!-- #endregion -->

Uma lista em Python é às vezes vista como equivalente de um *array* em outras linguagens, mas uma lista é mutável e pode conter elementos de tipos diferentes.

Quando uma array de fato contém elementos de um só tipo, temos algumas vantagens:
- Tamanho
- Desempenho
- Funcionalidades específicas


🎈 Link interessante: https://webcourses.ucf.edu/courses/1249560/pages/python-lists-vs-numpy-arrays-what-is-the-difference


#### Alguns exemplos

```python
n = 10_000_000
v = []
w = []
for i in range(n):
    v.append(i)
    w.append(i/10)
```

```python
type(v), type(w)
```

```python
%timeit v + w
```

```python
v_np = np.array(v)
w_np = np.array(w)
```

```python
type(v_np), type(w_np)
```

```python
v_np.dtype, w_np.dtype
```

```python
v_np.shape
```

```python
v_np.size
```

```python
v_np.nbytes/v_np.size
```

```python
%timeit v_np + w_np
```

#### E o módulo array?

```python
import array
```

```python
v_arr = array.array('d', v)
w_arr = array.array('d', w)
```

```python
v_arr.typecode
```

```python
%timeit v_arr + w_arr
```

🎈 Link interessante: https://medium.com/@gough.cory/performance-of-numpy-array-vs-python-list-194c8e283b65


#### Como isso acontece?

As arrays do NumPy são eficientes porque 
- são compatíveis com as rotinas de álgebra linear escritas em C/C++ ou Fortran
- são *views* de objetos alocados por C/C++, Fortran e Cython
- as operações vetorizadas em geral evitam copiar arrays desnecessariamente

```python
Image(url="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-020-2649-2/MediaObjects/41586_2020_2649_Fig1_HTML.png", width=600)
```

*Fonte: Harris et al., "Array Programming with NumPy", Nature volume 585, pages 357–362 (2020)*

<!-- #region tags=["section"] -->
## Vetorização
<!-- #endregion -->

Vetorização é a capacidade de expressar operações em arrays sem especificar o que acontece com cada elemento individual (em outras palavras: sem usar loops!)

- Código vetorizado é mais conciso e legível
- O código fica (ligeiramente) mais parecido com a notação matemática
- Mais rápido (usa operações otimizadas em C/Fortran)


### Vetorização: Exemplo 1

```python
v = np.array([1, 2, 3, 4])
```

```python
v**2
```

```python
np.power(v, 2)
```

```python
u = np.array([4.5, 3, 2, 1])
```

```python
np.dot(u, v) 
```

#### Nota: onde estão implementadas as funções do NumPy?

```python
print(np.dot.__doc__)
```

```python
np.info(np.dot)
```

```python
np.dot?
```

---


❓❓
- [Código fonte da função `np.dot`](https://github.com/numpy/numpy/blob/master/numpy/core/multiarray.py) - **linha 716**
- [Agora de verdade](https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/methods.c) - **linha 2269**
- [For realz](https://github.com/numpy/numpy/blob/master/numpy/core/src/multiarray/multiarraymodule.c) - **linha 981 / 1023**


### Cuidado!

```python
sum?
```

```python
sum(range(5), -1)
```

```python
from numpy import *
sum(range(5), -1)
```

```python
np.sum?
```

```python
%reset
```

```python
import numpy as np
```

---


#### Vetorização: Exemplo 2

```python
A = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8], 
              [9, 10, 11, 12]])
```

```python
A.T
```

```python
A.shape
```

```python
A.ndim
```

```python
A.reshape((2, 6))
```

```python
A.reshape((4, 3))
```

```python
A.ravel()
```

```python
u = np.random.rand(10)
u
```

```python
u.ndim
```

```python
u.shape
```

```python
u.T
```

```python
u.T.shape
```

```python
u.T == u
```

```python
u.T is u
```

```python
np.transpose?
```

```python
v = u.T
v.base is u
```

```python
u[0] = 0
```

```python
u
```

```python
v
```

[Documentação para ndarray.base](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.base.html)


#### ❓ (O que são essas dimensões?)

```python
from IPython.display import Image
Image(url="https://fgnt.github.io/python_crashkurs_doc/_images/numpy_array_t.png", width=600)
```

*[Fonte: Elegant SciPy](https://www.oreilly.com/library/view/elegant-scipy/9781491922927/ch01.html)*


#### Vetorização: Exemplo 3

```python
A[0, :]  
```

```python
A.sum()
```

```python
A.sum(axis=0)
```

```python
A.sum(axis=1)
```

```python
A.size * A.itemsize
```

```python
A.nbytes
```

Novamente:

```python
Image(url="https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41586-020-2649-2/MediaObjects/41586_2020_2649_Fig1_HTML.png", width=600)
```

<!-- #region tags=["section"] -->
### Broadcasting
<!-- #endregion -->

Permite fazer operações vetoriais de maneira generalizada.

☠️ **Cuidado!** ☠️

```python
x = np.array([1, 2, 3, 4])
```

```python
x + 5
```

```python
A
```

```python
x
```

```python
A+x
```

```python
np.add?
```

### 📒 Exercícios 


Como determinar a dieta mais econômica que satisfaz os requerimentos nutricionais básicos para boa saúde? Vamos supor que queremos fazer nossa compra semanal no supermercado: leite, miojo e café. Atualmente, os preços são:

- Leite: 4 reais o litro
- Miojo: 1,50 o pacote
- Café: 10 reais a unidade

Vamos nos preocupar com 4 ingredientes nutricionais básicos: proteína, carboidratos e gordura.

| Produto      | Carboidratos | Proteínas | Gorduras|
---------------|--------------|-----------|---------|
|Miojo (100g)  | 63,6g        | 10,0g     | 15,5g   |
|Leite (100ml) | 4,66g        | 3,32g     | 3,35g   |


#### Exercício 1

Use uma ndarray `a` bidimensional que armazene, na posição $(i, j)$, a quantidade de gramas do nutriente $i$ contidas em cada grama do alimento $j$.


#### Solução 1

```python
tabela = np.zeros((3, 2))
```

```python
miojo = np.array([63.6, 10, 15.5])
leite = np.array([4.66, 3.32, 3.35])
```

```python
# Para ter os valores nutricionais por grama (ao invés de para cada 100g)
miojo = miojo/100
# Mesma coisa para o leite: vamos obter o valor por ml
leite = leite/100
```

```python
tabela[:, 0] = miojo
```

```python
tabela
```

```python
tabela[:, 1] = leite
```

```python
tabela
```

#### Solução 2

```python
miojo = np.array([63.6, 10, 15.5])
leite = np.array([4.66, 3.32, 3.35])
```

```python
tabela = np.vstack([miojo, leite]).T/100
```

```python
tabela
```

#### Solução 3

```python
tabela = np.stack([miojo, leite], axis=1)/100
```

```python
tabela
```

#### Exercício 2


A partir da array `a` criada acima, extraia um array que contenha os valores de nutriente por alimento.

```python
carbs = tabela[0, :]
proteina = tabela[1, :]
gordura = tabela[2, :]
```

```python
carbs.shape, proteina.shape, gordura.shape
```

#### Exercício 3


Calcule o preço de uma compra incluindo 2 litros de leite, 1 pacote de café e 5 pacotes de miojo.

Para facilitar, vamos usar a seguinte notação:

$x_0$ -> quantidade de litros de leite;

$x_1$ -> quantidade de pacotes de miojo;

$x_2$ -> quantidade de pacotes de café;

Isso significa que podemos representar a quantidade a ser comprada como:

```python
x = np.array([2, 5, 1])
```

```python
# Leite: 4 reais o litro
# Miojo: 1,50 o pacote
# Café: 10 reais a unidade
c = np.array([4, 1.5, 10])
```

Total da compra:

```python
x[0]*c[0] + x[1]*c[1]+x[2]*c[2]
```

```python
total = np.dot(c, x)
total
```

O custo pode ser calculado, em geral, como

```python
def custo(x):
    c = np.array([4, 1.5, 10])
    return np.dot(x, c)
```

Assim, se mudamos as quantidades para

10 litros de leite; 8 pacotes de miojo; 2 pacotes de café

temos

```python
custo(np.array([10, 8, 2]))
```

---


### Operações vetorizadas e pontos flutuantes

```python
x = np.linspace(-np.pi, np.pi, 10)
```

```python
x
```

```python
np.sin(x)
```

#### ⚠️⚠️ Observação

```python
0 * np.nan
```

```python
np.nan == np.nan
```

```python
np.inf > np.nan
```

```python
np.nan - np.nan
```

```python
np.nan in set([np.nan])
```

Observe:

```python
np.array(0) / np.array(0)
```

```python
np.array(0) // np.array(0)
```

```python
np.array([np.nan]).astype(int) #.astype(float)
```

```python
0.3 == 3 * 0.1
```

**Nunca teste se um número real é igual a outro! Use sempre tolerâncias.**

```python
np.abs(0.3 - 3*0.1)
```

#### Explicação


- `NaN` : *Not a Number*
- `Inf` : Infinito?

```python
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
```

```python
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)
```

```python
np.finfo(np.float64).max+1.e308
```

Leia mais na documentação oficial do Python: https://docs.python.org/pt-br/3/tutorial/floatingpoint.html

```python
x = np.arange(0, 10*np.pi, np.pi)
```

```python
np.sin(x) == 0
```

```python
np.allclose(np.sin(x), np.zeros(np.shape(x)))
```

---


### Basic Indexing, Fancy Indexing

```python
x = np.arange(10)
x[2:5]
```

```python
x[:-7]
```

```python
x[1:7:2]
```

```python
y = np.arange(35).reshape(5,7)
y[1:5:2,::3]
```

As arrays do NumPy podem ser indexadas por outras arrays (ou qualquer outro objeto tipo-sequência que possa ser convertido para uma array, como listas, com a exceção de tuplas). No entanto, isso pode não ser tão eficiente pois ativa o [indiciamento avançado](https://numpy.org/devdocs/reference/arrays.indexing.html#advanced-indexing).

```python
x = np.arange(10,1,-1)
```

```python
x[np.array([3,3,-3,8])]
```

```python
x[np.array([[1,1],[2,3]])]
```

Também podemos usar máscaras booleanas:

```python
b = y>20
```

```python
y[b]
```

```python
x = np.arange(30).reshape(2,3,5)
```

```python
b = np.array([[True, True, False], [False, True, True]])
x[b]
```

```python
x = np.arange(5)
x[:,np.newaxis] + x[np.newaxis,:]
```

```python
z = np.arange(81).reshape(3,3,3,3)
z[1,...,2]
```

```python
z[1,:,:,2]
```

### Exercícios


#### Criar um array aleatório de tamanho 12 e substituir o menor valor por -999.

```python
aleatorio = np.random.random(12)
aleatorio[aleatorio.argmax()] = -999
aleatorio
```

#### Ordenar uma array pela n-ésima coluna

```python
vetor = np.random.randint(0,10,(3,3))
vetor
```

```python
vetor[vetor[:,1].argsort()]
```

#### Como saber se uma array 2D tem colunas nulas?

```python
z = np.random.randint(0,3,(3,10))
z
```

```python
(~z.any(axis=0)).any()
```

```python
z[:, 0] = np.zeros(3)
z
```

```python
(~z.any(axis=0)).any()
```

#### Trocar duas linhas de uma matriz

```python
A = np.arange(25).reshape(5,5)
A
```

```python
A[[0,1]] = A[[1,0]]
A
```

### 2.4 Submódulos

- `numpy.random` 
- `numpy.fft`
- `numpy.ma`
- `numpy.linalg`
- `numpy.f2py`

```python
print(np.__doc__)
```

---


[Voltar ao notebook principal](00-Tutorial_Python_Brasil_2020.ipynb)

[Ir para o notebook Matplotlib](02-Tutorial_Matplotlib.ipynb)

```python

```
