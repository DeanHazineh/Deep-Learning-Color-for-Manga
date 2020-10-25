# Manga Colorization via cGAN for Deep learning
For a general discussion of the work, see the pdf in the repository which has a complete discussion of the concept. 


## Overview of the task
The problem of colorizing with and without color hints

<img src=/images/MainProblemExample.png alt="drawing" height="500" width="500"/><img src=/images/ModelSchematic.png alt="drawing" height="500" width="400"/>

Generator Model

<img src=/images/GeneratorModel.png alt="drawing" width="600"/>

Discriminator Model

<img src=/images/DiscriminatorModel.png alt="drawing" width="600"/>


## Creating Training Data
A single data-point for fitting/training the model requires three components: (1) a high-quality, digitally colorized anime/manga-esque image, (2) a corresponding binary sketch version of that image, and (3) (in the case of the color cues experiment) a corresponding color cue that I can imagine an artist would provide the hypothetical generator in order to get out the real, digitally colorized image. In order to make this project realizable, each of these three components must be quickly obtained and/or generated in a fully automated process.

The first step in the training data generation process is to obtain a large set of digitally colorized images that fit the style of Japanese manga and anime This is actually easily done by writing a web-scraping python script that automatically searches and downloads tagged images from the imaging hosting website, Danbooru [Danbooru URL\cite{DanbooruSite}]. This site is ideal for this task since it is essentially a large-scale crowdsource and tagged anime dataset with nearly 4 million anime images (and reportedly over 108 million image tags total allowing quick filtering and searching. Fair Warining: many of the images on this site are arguably NSF content. Using the code on the github, I downloaded approximately 9000 images (at about 1 second per image) with fantasy themed tags like "holding staff" and "magic solo". After, these images are further processed in python via resizing and cropping such that all images are converted to a 256 x 256 square. This then satisfies component 1 making up the high-quality, digitally colorized manga-esque art.

The corresponding sketches and color-cues are then obtained by automatic image processing on these downloaded, colored images. In order to obtain the black-and-white sketches (again devoid of effects like shading since I want to reduce drawing time for artists) from the digitally colorized images, I apply an edge-detection algorithm via OpenCV which implements Gaussian-weighted adaptive thresholding to binarize the image. Here, there is freedom of choice to decide what blocksize to use in the algorithm, i.e. the number of neighboring pixels to sum over when thresholding, and different values produce a slightly different appearance to the "sketchified" image. An example of a colored image and the corresponding sketches resulting from this method with different blocksize parameterizations is displayed. Throughout this work, I utilize a blocksize of 7 since it provides some distinction between primary edges and finer details as if the artist had utilized two different pencil sizes which seems realistic. Above these images, there is also displayed a "complexity value" for the sketch which is utilized in a system to automatically identify images for which this sketch process does not work well by virtue of the colored images having too many edges or abnormal global color gradients. The complexity value is simply computed by  calculating the percent of all pixels that has been set to 0 (the black edges). From inspection of samples in the data set, I discard all images with a complexity value greater than 40 at blocksize characterization 7 (i.e greater than 40\% of the pixels depicting an edge).

<img src=/images/CompareSketchify.png alt="drawing" width="600"/> <img src=/images/TrainingData.png alt="drawing" width="350"/>

For the last step in the training data generation process, I establish a method to automatically derive color cues corresponding to the colored images. I imagine an effective system would be for an artist to very quickly create color cues by drawing over the sketch with large highlighters of different colors. In the imagined process, they would also need not worry about coloring within the lines or specifying variations in color or shade. To approximate this in a way that could be sufficient for a proof-of-concept implementation and without requiring any manual dataset labeling, I create the color cues for each image by spatially blurring the colored images with a large Gaussian kernel. From testing, I found that a Gaussian filter with a standard deviation of 20 pixels qualitatively produces the desired effect. For reference, this corresponds to a FWHM of approximately 50 pixels which is nearly 1/5 the width of the image. Three examples of the color cues derived from this method and their corresponding colored images are shown.

<img src=/images/ColorCueGeneration.png alt="drawing" height="600" width="400"/>


## Example Results With and Without Color Cues
<img src=/images/NoColorCuesResults.png alt="drawing" width="600"/>
<img src=/images/WithColorCuesResults.png alt="drawing" width="600"/>
