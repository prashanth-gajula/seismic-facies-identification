# seismic-facies-identification

Seismic facies analysis is one of the most difficult tasks in the oil and gas sector, and it requires a lot of time and human resources to complete. By automating the process with deep learning techniques, both time and human effort will be reduced. Traditional Unet, Attention Unet VGG16 as the backbone, Unet Resnet18 as the backbone, and Unet Inceptionv3 as the backbone were used to solve this multi-class segmentation problem in this project. Our models are evaluated using Mean Intersection Over Union as a measure. All other models, with the exception of the classic Unet model, are learned using pre-trained imagenet weights. Traditional Unet performs the best out of all the models, with an average Iou of 0.87.

# Data:

data source and credit: https://www.aicrowd.com/challenges/seismic-facies-identification-challenge

The seismic facies analysis uses the 3D images which are created by collecting the echoes of sound waves which are sent to the ground, generally, the 3D seismic images cover 25x25 horizontal extent and 10km depth and facies are interfaces delineated by echoes in seismic images often mark boundaries between regions containing sediments deposited by different types of geological processes.

3D seismic image from a public-domain seismic survey called "Parihaka," accessible from the New Zealand government, categorized each pixel in the image into one of six distinct facies based on patterns done by Expert geologists.

The dataset is a three-dimensional image encoded as an array of 1006 x 782 x 590 real integers recorded in the order they appear in the image (Z, X, Y). The labels are similarly organized in a 1006 x 782 x 590 array of integers with values ranging from 1 to 6. Each integer label represents a categorization of each picture pixel into one of six possible facies.

The following are the geologic descriptions of each label:

1. Basement/Other: Basement - Low S/N; Few internal Reflections; May contain volcanic in places
2. Slope Mudstone A: Slope to Basin Floor Mudstones; High Amplitude Upper and Lower Boundaries; Low Amplitude Continuous/Semi-Continuous Internal Reflectors
3. Mass Transport Deposit: Mix of Chaotic Facies and Low Amplitude Parallel Reflections
4. Slope Mudstone B: Slope to Basin Floor Mudstones and Sandstones; High Amplitude Parallel Reflectors; Low Continuity Scour Surfaces
5. Slope Valley: High Amplitude Incised Channels/Valleys; Relatively low relief
6. Submarine Canyon System: Erosional Base is U-shaped with high local relief. Internal fill is a low amplitude mix of parallel inclined surfaces and chaotic disrupted reflectors. Mostly deformed slope mudstone filled with isolated sinuous sand-filled channels near the basal surface.

<img src='https://github.com/JayakeshavB/seismic-facies-identification/blob/main/images/Screen%20Shot%202022-05-22%20at%203.09.26%20PM.png'></img>

# Data Pre-Processing:

To do data normalization, we must first understand some of the image's properties, which include Mean (0.67661), and Standard deviation (390.30884), Min_Pixel - Max_Pixel (5195.5234 - 5151.7188).  

#### Min-Max Scaling:

Then, using the min and max pixel values as a guide, we performed Min-Max normalization on pixel values to make values ranging from -1 to 1.

#### Patches Generation:

The data was in a 3D image, therefore we had to convert it to 2D image patches in order to use it for modeling. For this project, we patchify the library, which creates 1006 2D patches of size  512x512 both images and corresponding masks.

<img src='https://github.com/JayakeshavB/seismic-facies-identification/blob/main/images/Screen%20Shot%202022-05-22%20at%203.10.05%20PM.png'></img>

#### Stacking The Patches:

For modeling, we stack all of the image patches into a single NumPy array, similar we stack all of the mask patches into a single array.

#### Label Encoding:

With the help of label encoding, we convert our pixels from [1,2,3,4,5,6] to [0,1,2,3,4,5]. Because we must transform all of our pixels into an ordered range, regardless of their value (35, 53, etc.). This allows us to use the predefined Iou function as a metric from Keras during model training.


# Feature Engineering:

#### One-Hot Encoding:

Our training data is more useful and expressive as a result of one-hot encoding, and it can be rescaled simply. We can more readily establish a probability for our values if we use numeric numbers. For our output values, we choose one-hot encoding in particular because it delivers more complex predictions than single labels. And we need one hot encode label when we are using deep learning models

# Exploratory Data Analysis

<img src='https://github.com/JayakeshavB/seismic-facies-identification/blob/main/images/Screen%20Shot%202022-05-22%20at%203.10.36%20PM.png'></img>

# Model Development:

1. Traditional Unet

2. Attention Unet VGG16 as the backbone

3. Unet Resnet18 as the backbone

4. Unet Inceptionv3 as the backbone 

This are the 4 models utilized to tackle this multi-class segmentation problem. Data augmentation is done on every model.

# Training Details:

#### Loss: categorical_focal_loss

#### Optimizer: Adam

#### Epochs: 50

#### Metric: MeanIOU

#### Class Weights: [ 1.01833657,  0.37961082,  3.34285935,  0.61222313, 18.06211106,  2.52634665]

# Results

### UNET

<img src='https://github.com/JayakeshavB/seismic-facies-identification/blob/main/images/Screen%20Shot%202022-05-22%20at%203.11.33%20PM.png'></img>

### ATENTION UNET

<img src='https://github.com/JayakeshavB/seismic-facies-identification/blob/main/images/Screen%20Shot%202022-05-22%20at%203.12.05%20PM.png'></img>

The modeling ipynb file contains the reaming model results.

# Conclusion 

In this study, we proposed a U-Net and a couple of variations of Unet with transfer learning models as the backbone and self-attention-based unet to perform automatic seismic facies analysis. We compared four models of different combinations  of convolution and attention mechanisms. The proposed unet with the attention model generates superior classification results over the other methods.


# Files:

=> Data Pre-processing File: seismic-facies-identification_data_pre_processing.ipynb

=> Modeling File: seismic-facies-identification_modeling.ipynb
