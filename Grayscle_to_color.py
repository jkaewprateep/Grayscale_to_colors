import tensorflow as tf
import tensorflow_io as tfio

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

file = "F:\\Pictures\\actor-Ploy\\119282942.jpg"
image = tf.io.read_file( file )
image = tf.io.decode_image( image, channels=0, dtype=tf.dtypes.int32, name="input_image", expand_animations=True )

grayscale_image = tf.image.rgb_to_grayscale( image )

cropped_image = image_1 =  tf.image.crop_to_bounding_box(image, 0, 0, 50, 50)
cropped_grayscale_image = image_2 = tf.image.rgb_to_grayscale( cropped_image )

print( cropped_image.shape )

cropped_image_width = int( cropped_image.shape[0] )
cropped_image_height = int( cropped_image.shape[1] )

image = tf.keras.preprocessing.image.array_to_img( image )
grayscale_image = tf.keras.preprocessing.image.array_to_img( grayscale_image )
cropped_image = tf.keras.preprocessing.image.array_to_img( cropped_image )
cropped_grayscale_image = tf.keras.preprocessing.image.array_to_img( cropped_grayscale_image )


print( cropped_image_width )
print( cropped_image_height )

dataset = { "RGB": [], "BW": [] }

icount = 0
height = 0
width = 0

for i in range( cropped_image_width ) :
	for j in range( cropped_image_height ) :

		dataset["RGB"].append( str( image_1[width][height][0].numpy() ) + "_" + str( image_1[width][height][1].numpy() ) + "_" + str( image_1[width][height][2].numpy() ) )
		dataset["BW"].append( image_2[width][height][0].numpy() )

		height = height + 1
	height = 0
	width = width + 1

data = tf.constant([ dataset["BW"] ])
layer = tf.keras.layers.IntegerLookup()
layer.adapt(data)
vocab = layer.get_vocabulary()

print( "===============" )

layer = tf.keras.layers.IntegerLookup(vocabulary=vocab)
image_3 = tf.zeros([ 50, 50, 3]).numpy()

height = 0
width = 0

icount = 0

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

plt.figure(figsize=(1, 4))
plt.title("Colors mapping")

plt.subplot(1, 4, 1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( image )
plt.xlabel( "Original" )

plt.subplot(1, 4, 2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( cropped_image )
plt.xlabel( "Cropped image" )

plt.subplot(1, 4, 3)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( cropped_grayscale_image )
plt.xlabel( "GrayScales image" )

plt.subplot(1, 4, 4)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( image_3 )
plt.xlabel( "Re-colors" )
	
plt.show()