# Grayscale_to_colors

Convert grayscales image back to colors, there are many discussion about recovers of the image data from previous lossy method or path fullfilled of the image from previous data or database patterns. Consider the gary_scale image is a lossy image created from 'tf.image.rgb_to_grayscale()' had result as [ width x height x 1 channel ] and original or target recovery has data [ width x height x 1 channel ].

## Load an image ##    

Using Tensorflow load image and decode as jpg, scales as gray image from ( width x height x channel ) to ( width x height x 1 channel ), some decoder from the experiment set required you setup the correct ski-image and python cache and sometimes you may found error using experiment function. The experiment functions is avioded to use in production environments and it can be substitude by updated of the external program.
```
file = "F:\\Pictures\\actor-Ploy\\119282942.jpg
image = tf.io.read_file( file )
image = tf.io.decode_image( image, channels=0, dtype=tf.dtypes.int32, name="input_image", expand_animations=True )
grayscale_image = tf.image.rgb_to_grayscale( image )
``` 

## Crop an image ##  

Using Tensorflow crop an image, convert to image object to next presentation layer. The result from tf.image.crop_to_bounding_box( ) has 4 dimensions because they are continouse process and operation with the array sizes [ 1, width , height, channel ] possible to many of target function output matrixes. The array to image conversion is required because we do operations on the image input and sometimes those inputs are from many sources and scales ploting with out standardized of data may see black or white colors image output and none scales data are estimate. Power of calculation saved by removed unranges data.
```
cropped_image = image_1 =  tf.image.crop_to_bounding_box(image, 0, 0, 50, 50)
cropped_grayscale_image = image_2 = tf.image.rgb_to_grayscale( cropped_image )
cropped_image_width = int( cropped_image.shape[0] )
cropped_image_height = int( cropped_image.shape[1] )

image = tf.keras.preprocessing.image.array_to_img( image )
grayscale_image = tf.keras.preprocessing.image.array_to_img( grayscale_image )
cropped_image = tf.keras.preprocessing.image.array_to_img( cropped_image )
cropped_grayscale_image = tf.keras.preprocessing.image.array_to_img( cropped_grayscale_image )
```

## Create dataset ##  

Using simple loop to extract each pixel into dataset, can be use as model dataset later. Preparing input data from original data and grap_scale conversion result data, it is only dataframe or temporary table data mapping but gray_scale converted image is losses recovered from loss contast image is scales functions but the significant data revoery is shape and backgrounds. 

``` dataset = { "RGB": [], "BW": [] } 
for i in range( cropped_image_width ) :
	for j in range( cropped_image_height ) :

		dataset["RGB"].append( str( image_1[width][height][0].numpy() ) + 
          "_" + str( image_1[width][height][1].numpy() ) + 
          "_" + str( image_1[width][height][2].numpy() ) )
		dataset["BW"].append( image_2[width][height][0].numpy() )

		height = height + 1
	height = 0
	width = width + 1
```

## Create Lookup Vocaburary ##  

Using Tensorflow LookUp create Lookup table or vocabrary, the vocaburary is unique key and index and the duplicated key results ambigious result return, it required to be unique key.

```
data = tf.constant([ dataset["BW"] ])
layer = tf.keras.layers.IntegerLookup()
layer.adapt(data)
vocab = layer.get_vocabulary()
```

## Create re-colors image from gray-scale image ##  

Using Tensorflow create new image from grayscale colors to colors, mapping of the gary scale image color to the vocaburary or the temporary table and normalized for the results.

```
layer = tf.keras.layers.IntegerLookup(vocabulary=vocab)
image_3 = tf.zeros([ 50, 50, 3]).numpy()

for i in range( cropped_image_width ) :
	for j in range( cropped_image_height ) :
	
		data = tf.constant([ image_2[width][height].numpy() ]).numpy()
		temp = list(dataset.values())[0][int(layer(data).numpy())].split("_")
		
		print( temp )
		
		Red = temp[0]
		Green = temp[1]
		Blue = temp[2]
		
		image_3[width][height][0] = int(Red)
		image_3[width][height][1] = int(Green)
		image_3[width][height][2] = int(Blue)

		height = height + 1
	height = 0	
	width = width + 1
  
image_3 = image_2 - image_3
image_3 = tf.keras.layers.Normalization(mean=3., variance=2.)( image_3 )
image_3 = tf.keras.layers.Normalization(mean=4., variance=6.)( image_3 )
image_3 = tf.keras.layers.Conv2D(3, (3, 3), activation='relu')( tf.expand_dims( image_3, axis=0 ) )
image_3 = tf.keras.preprocessing.image.array_to_img( tf.squeeze( image_3 ) )
```

## Result image ##
![Alt text](https://github.com/jkaewprateep/Grayscale_to_colors/blob/main/Figure_6.png?raw=true "Title")
