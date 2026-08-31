# Red de clasificacion de imagenes

### Conceptos

- Como se puede clasificar una imagen con una red neuronal
  tengamos en cuenta que introducir una imagen en una red neuronal es un proceso distinto a un simple numero. en este caso debemos tomar la imagen necesaria y clasificar cada pixel a un valor numerico asociado (0 es totalmente negro y 255 es totalmente blanco). esto para cada uno de los pixeles de la imagen. y por cada pixel tenemos una neurona. el cual si la imagen es de 100x100 tiene 10,000 neuronas. esto debemos minimizarlo lo mas posible. ya que pues usar 10,000 neuronas apenas en la capa de entrada va a ser extremadamente profunda nuestra red.
  para nuestro caso vamos a usar imagenes de 28x28 y 10 neuronas de salida (10 clases).

- Que tipo de red neuronal vamos a usar?: para este tipo de problemas hay una llamada red neuronal convolucional. pero vamos a empezar con una red densa. y luego iremos descubriendo cosas. una red neuronal lineal. solo puede resolver problemas (ecuaciones )lineales. para que las redes puedan resolver problemas mas complejos se usan herramientas como (capas ocultas, funciones de activacion)

* las capas ocultas sirven para que haya mas espacio para transformaciones en la red. esto permite a la red ser mas configurable. sin embargo las capas ocultas por si solas aun no son capaces de resolver problemas mas complejos (no lineales
  ).

* las funciones de activacion sirven para que la red pueda aprender mas complejos patrones. hacen que nuestra red no sea lineal el cual recibe los valores ponderados de la entrada y su peso mas el sesgo y retorna el valor de la salida.

ten en cuenta que el proceso de que una red sea eficiente es cuestien de probar y comparar resultados. buscando la mejor arquitectura de red para nuestro problema.

para entrenar una red necesitamos muchos datos de ejemplos para que vaya asimilando cada una de las clases.si el problema es mas complejo puede que necesitemos mas datos.
