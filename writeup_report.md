** Project 3: Behavior Cloning **

The goal of the project is to gather data, build and train a steering wheel model and run a simulation of self driving car.

** Results **

- I used data provided by Udacity and also data that captured by simulator. Final model was trained with data provided by Udacity.
- Model of convolutional neural network was built and trained using Keras frontend to TensorFlow framework. Input of CNN are images from central, left and right cameras and output is prediction of steering wheel angle
- To accommodate large training data set, model was rewritten using generator for training and validation data source
- Video of simulation of track 1 posted on youtube [https://youtu.be/7Nv06lXlp7o]
- Model currently fails second track. Additional data and image normalization required.

** Architecture **

Model consist of one lambda layer, 4 layers of convolutional neural network and 2 fully connected layers
1. Lambda layer utilized to normalize input images i.e. convert RGB integer encoding from range [0..255] to float encoding with range [-1, 1]. I found it very useful since lambda layer transferred as a part of model and allows to avoid image preprocessing during testing of the network. Total number of signals is 160x320x3
2. First convolutional layer has 5x5x3 input and outputs to 6 channels and also has strides size 4. Setting strides to value more then 1 make possible to avoid MaxPool layer. Total number of the output signals is 40x80x6. To improve generalization Dropout added to first layer
3. Second convolutional layer also has 5x5 input and outputs to 12 channels. Stride size is 2. Total output is 20x40x12
4. Third convolutional layer has smaller input 3x3 and outputs to 24 channels. Stride size is 2. Total output is 10x20x24
5. Last convolutional layer has 3x3 kernel and outputs to 40 channels. Total number of output neurons are 5x10x40 => 200
6. First fully connected layer has 80 neurons, also it utilizes Dropout for better generalization
7. Output layer has 1 neuron and also uses Dropout.

I've tried multiple alternative architectures and found this one satisfactory. It learn data rather quickly and does not demonstrate signs of overfitting.

** Training **

To train network "adam" optimizer was selected and "mean square error" loss function selected. Adam optimizer selected because it has embedded optimization of momentum and supports decay of learning rate

I selected to use batch size of 32 mostly due to memory limitation on my laptop. Initially, I used AWS GPU instances and it can consume entire dataset of 15K images without batching.

Even I used 10 epochs for training, model does not demonstrate significant improvements after epoch 4. I use 10 to check that model is not overfitting, i.e. small change in loss on training set, follows with small change of loss on validation set.

** Data **

After multiple attempts to capture my own data and use various architectures I decided to use dataset that provided by Udacity. I'm simply bad keyboard driver.

There are 2 misbalances in original dataset:
1. "straight" areas of the track dominate in data set.
2. Track has more left turns.

So, data has to be rebalanced. Otherwise, even with great training and validation results, model has tendency not to turn when required. To balance out such areas I've made next adjustments:
1. Reduce number of samples with angle equals 0 to 10%
2. Include images from right and left cameras with steering angle adjustment.
3. Include vertically flipped images

To summarize, images from all 3 cameras are used:

center
![ALT center](center_2016_12_01_13_31_14_500.jpg)  

left:
![ALT left](left_2016_12_01_13_31_14_500.jpg)

right
![ALT right](right_2016_12_01_13_31_14_500.jpg)

** Genesis **

As soon as I finished with normalization of the data set balance it became obvious that model does not need to be complicated. Original model had more hidden fully connected layers, but multiple reduction demonstrated that it is not required.

Also, addition of Dropout to convolutional layer have not demonstrate better generalization. So, I've remove Dropout and replaced MaxPooling with simple convolutional and strides.

Also experiments show that hidden layer with 80 neurons and Dropout is better for generalization then layer with 120 and without Dropout.
