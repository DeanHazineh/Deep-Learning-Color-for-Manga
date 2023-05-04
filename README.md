# Deep Learning Color for Manga and Anime Sketches

## [Read the PDF write-up for this project here](Deep_Learning_Color_for_Manga_and_Anime_Sketches.pdf)

## Overview: The Problem of Colorizing With and Without Color Hints
<div style="display:flex">
  <img src="/images/ModelSchematic.png" alt="Alt Text 1" width="31%">
  <img src="/images/MainProblemExample.png" alt="Alt Text 2" width="46%">
</div>

In this work, the goal was to explore if deep learning can be employed
in order to enable Manga artists to create colorized Manga prints
instead of the traditional black-and-white prints without requiring
significant extra time or work for the creator and without requiring a
compromise to the mediums style. While much of the results shown clearly
display the results of the investigation into machine colorization, the
question of sketch enhancement is also subtly probed as I seek to create
an architecture which not only colorizes an image but starts with a
unpolished, early stage sketch devoid of quality shading. It would be
considered a success in my opinion if one can begin with a less final
starting image (reducing drawing time for the artist) and still then
achieve both style preservation, feature finalization and full
colorization by implementation of our proposed architecture. Here, I
highlight that I do not believe an un-supervised colorization approach
makes sense as a real-world solution for this particular problem, and as
a result, I have designed the code focusing on a proof-of-concept
solution that allows the artist to give input via the “color highlight”
used for the overall image. With that being said, I have also generated
results and a study excluding the artist input entirely (setting color
inputs to a zero-matrix and retraining the model) in order to
investigate in an academic nature what the results would be. I think both are equally interesting and amusing investigations
worth presenting.

Here, I find that the best solution/starting point for this task is to
implement a conditional adversarial network with a CNN architecture and
to focus predominantly on framing the problem as an image-to-image
translation task. The motivation and guidance for which our
implementation is closely built from is the famous, so-called Pix2Pix
architecture released in late 2016 by A. Efros’s team at the Berkley AI
Research Lab. 

### Deep Learning Model
The model for the generator and discriminator used in this work is the
same as that reported in the Pix2Pix paper with a
review of the technical details summarized in the PDF.  Both the generator and discriminator utilizes
modules of the form convolution-BatchNorm-ReLu. For the related code
assembling the generator and discriminator, see the scripts “mainv3.py".

### Creating Training Data
The first step in the training data generation process is to obtain a
large set of digitally colorized images that fit the style of Japanese
manga and anime. This is actually easily done by writing a
web-scraping python script that automatically searches and downloads
tagged images from the imaging hosting website, Danbooru. This site is ideal for this task since it is
essentially a large-scale crowdsource and tagged anime dataset with
nearly 4 million anime images (and reportedly over 108 million image
tags total allowing quick filtering and searching). Fair Warining: many of the images
on this site are arguably not “safe-for-work” content. Using the code
released on my github, I downloaded approximately 9000 images (at about
1 second per image) with fantasy themed tags like “holding staff” and
“magic solo”. After, these images are further processed in python via
resizing and cropping such that all images are converted to a 256 x 256
square. This then satisfies component 1 making up the high-quality,
digitally colorized manga-esque art.

For the last step in the training data generation process, I establish a
method to automatically derive color cues corresponding to the colored
images. I imagine an effective system would be for an artist to very
quickly create color cues by drawing over the sketch with large
highlighters of different colors. In the imagined process, they would
also need not worry about coloring within the lines or specifying
variations in color or shade. To approximate this in a way that could be
sufficient for a proof-of-concept implementation and without requiring
any manual dataset labeling, I create the color cues for each image by
spatially blurring the colored images with a large Gaussian kernel. From
testing, I found that a Gaussian filter with a standard deviation of 20
pixels qualitatively produces the desired effect. For reference, this
corresponds to a FWHM of approximately 50 pixels which is nearly 1/5 the
width of the image. Three examples of the color cues derived from this
method and their corresponding colored images are shown in below:

<img src=/images/ColorCueGeneration.png alt="drawing" height="600" width="400"/>

### Example Results With and Without Color Cues

In order to evaluate the capabilities of the trained models for both
experiments in a fair and insightful way, I review the performance of
each on the same, carefully selected set of testing images. Six of these
images can be seen in the figures following. These six images were chosen as they
collectively present three different levels of difficulty. As such, I
have classified the six into three groups referred to in this work as
evaluation “tasks”. A summary of the anticipated challenge each task
presents is reviewed following, listed in increasing order of
difficulty:

-   The Color Task:\
    For images in this class, the model has trained on several other
    images directly containing these characters although presented in a
    different pose or drawn by a different artist. This will test the
    performance for cases where the model is well-conditioned to
    formulate the problem as a classification task and colorize with the
    aid of object/character recognition.

-   The Transfer Task:\
    No direct variation of the characters present in these test images
    have been previously shown to the model during training; however,
    the overall theme and complexity level of images in this task
    closely match that of the training set. This test will evaluate how
    well the model can perform true image translation with limited
    reliance on character memorization.

-   The Interpolation Task:\
    These images have complexity values well over 20% higher than the
    cutoff threshold defined for the training data set. The model has
    also never seen any variation of these characters, this type of
    drawing style (it no longer has features similar to the downloaded
    Danbooru anime set), or images with a similar density of edges.

The results of the three aforementioned evaluation tasks are here
displayed in Figure [fig:NoColorCuesResults] for the colorization
experiment where no artist color cues are provided to the generator and
in Figure [fig:WithColorCuesResults] for the colorization experiment
where color cues are provided. For emphasis, the former can be
classified as the unsupervised deep-learning approach while the latter
is the supervised.

<div style="display:flex">
  <img src="/images/NoColorCuesResults.png" alt="Alt Text 1" width="46%">
  <img src="/images/WithColorCuesResults.png " alt="Alt Text 2" width="46%">
</div>
