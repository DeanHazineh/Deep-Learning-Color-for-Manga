# Manga Colorization via cGAN for Deep learning
For a general discussion of the work, see the pdf in the repository which has a complete discussion of the concept. 

## Overview of the task
The problem of colorizing with and without color hints
<img src=/images/MainProblemExample.png alt="drawing" height="500" width="500"/><img src=/images/ModelSchematic.png alt="drawing" height="500" width="400"/>

Examining the literature, one can find that there has actually been a number of academic investigations into the question of deep-learning color for black and white images. There has in fact even been some exploration by researchers directly into the problem of deep-learning color for Manga itself [cGAN-based Manga Colorization \cite{GANColorization}]; however, I found this work and others to be largely incomplete/inconclusive or, in my opinion, flawed regarding a realistic extension to industry and use by artists. From my search, all previous works have explored the topic of unsupervised deep-learning for colorization, which as noted before is not the primary implementation I have in mind, but since they serve as a backdrop to my architecture and one of the side-explorations conducted here, I first briefly review the recent developments on this topic.\\

Although outside the context of Manga, a great and relevant introduction to the topic of unsupervised, deep-learning colorization is given by Richard Zhang et al at Adobe Research [Group Github, \cite{ColorPhotographs}]. By training a feed-forward pass CNN on over one-million images from the imagenet database, their group published in 2016 the then state-of-the-art architecture for converting gray-scale photographs to vibrant colored images. He has aptly described this problem of general, unsupervised colorization as "the problem of hallucinating a plausible color version of the photograph", but this is immediately seen to be a severely under-constrained inverse problem necessitating some form of support [Colorful Image Colorization Paper \cite{Colorful Image Colorization}]. By modifying the standard CNN algorithm (with some changes outside the context of this work) and with enough training data, the group demonstrated that highly realistic colorization can be achieved under the context of re-framing the problem as a large classification task. The performance is then found to be constrained largely by the usual "data-set feature bias". This and previous works therein cited provide guidance that a CNN can be an effective architecture for learning color thus encouraging its use as a starting point in our task.\\

With that said, however, the previously cited architecture by the Zhang team in its current form would be insufficient for this task without changes for two key reasons. First, the Manga training data set which is here created through a careful but automated process feasibly results in a set (at this current time) on the order of tens of thousands of images instead of millions of images like in the cited work. A full re-framing of our problem as a classification task where shapes are associated to colors, even in the supervised color hint scenario, appears unlikely to be effective\footnote{At this time, I am uninterested in exploring transfer learning from the imagenet set given a better alternative.}. Second and more importantly, their work and similar work in the field at that time (around 2016) implemented the Euclidean L2 loss between the ground truth and predicted color as the objective function. Their work has shown that without some complex randomization, this loss function favors grayish colors that leads to desaturated results in total contrast to the typical color palletes in Manga/Anime art-style. Furthermore, it has been well shown in recent years that this MSE type loss is largely ineffective for image translation tasks since it leads to blurry outputs and poor contrast due to an effective averaging of all possible outputs [Summary and note of papers \cite{MSE Loss Blur}]. Without using an L2 Euclidean loss, I do not a priori know what is the best loss function is for this task and for this reason, I turn (with ample motivation in literature) to implementation of an adversarial loss. In theory, a GAN should not suffer from the mean image problem at all by virtue of optimizing an entirely different type of mathematical divergence.\\

Here, I find that the best solution/starting point for this task is to implement a conditional adversarial network with a CNN architecture and to focus predominantly on framing the problem as an image-to-image translation task. The motivation and guidance for which our implementation is closely built from is the famous, so-called Pix2Pix architecture released in late 2016 by A. Efros's team at the Berkley AI Research Lab [Pix2Pix Github \cite{Pix2Pix Github}]. To summarize, an image-to-image translation task is the problem of translating one possible representation of a scene to another given sufficient training data. In the past, this has required customized and hand-engineered objective functions as seen in the paper from Richard Zhang's group; however, the exploration underlying the Pix2Pix implementation was to remove that step by creating a general image translation architecture which could work across different problems and to demonstrate that a conditional generative adversarial network (cGAN) can effectively self-learn the required loss function for many different image tasks [Image-to-Image Translation Paper \cite{Pix2Pix Paper}]. Again, coming up with the loss-function analytically is not only an open-research problem but it is a non-trivial task generally requiring expert knowledge since the standard L1 and L2 Euclidean distance minimization leads to blur. The particular details of the model used in this work is left to the Methodology section with rigorous theoretical treatment deferred to the original Pix2Pix paper [Image-to-Image Translation Paper \cite{Pix2Pix Paper}]; however, here I present a high-level summary of the architecture's unique and key features.\\

The two fundamental components of the GAN-based architecture are the generator and the discriminator--two separate neural networks that are each individually optimized in a feedback loop. The two networks play a role analogous to that of an art forger and an art critic and are pitted against each other in a zero-sum game where the generator is tasked with creating a fake painting to fool the critic (the machine generated colorized Manga) and the discriminator is tasked with identifying the fake art from the original art (the artist colorized Manga in the training set). The training process for the GAN entails that the generator actively learns what group-features comprise the original art and gets better at reproducing it while the discriminator simultaneously learns new tricks to distinguish and identify the forged art. In summary, the GAN learns a numeric loss function that classifies an output image as real or fake while simultaneously training a generative model that minimizes this loss. In the cited work and in my work here, I utilize an objective function that is actually a combination of the GAN loss with a L1 loss along with slight modifications to a conventional CNN generator and discriminator as discussed in the following section

In summary, the goal of the project is to create a mapping that takes an artist's sketch and any color specifications and turns it into a final colorized image with the intended style preserved. Specifically, the input passed in to the model is a {256 x 256 x 1} binary image encoding the black-and-white initial sketch in addition to a {256 x 256 x 3} RGB image encoding a hint/instruction regarding what general color scheme the artist wants the AI model to use when colorizing the image (this is referred to throughout as the color cue). The output would then be a {256 x 256 x 3} image that is both a colorized and feature enhanced rendition of the black and white sketch originally given to the generator. Here, I study the quality of the output image in two cases-- (A) when no color cue is provided such that the color cue matrix is initialized to an array of ones for all sketches and (B) when color cues are given. These two experiments are pictorially diagrammed in Figure. \ref{fig:ModelSummary}. For each experiment, the discriminator and generator are trained for 30 epochs on an NVIDIA GeForce RTX 2070 Mobile GPU unit (7 GB memory) which takes approximately 2.7 hours for either case. The set of three images shown in in Figure \ref{fig:ModelSummary} for Experiment A and B accurately depict the type of images used in the training set for each experiment respectively. For both cases, the training data images are the same excluding the obviously different color cue matrices. The training data used for this report consists of approximately 9000 unique sets (i.e. 9000 triplets of sketch + color cues + artist colorized images). The testing data set includes images hand-picked by myself which stress-tests the trained model in various ways and has never been previously shown to the generator during training.   

## Deep Learning Model
The model for the generator and discriminator used in this work is the same as that reported in the Pix2Pix paper \cite{Pix2Pix Paper} with a review of the technical details summarized here. For a quick visualization, a schematic of the generator is shown in Figure \ref{fig:generator} and that of the discriminator is shown in Figure \ref{fig:discriminator}. Both the generator and discriminator utilizes modules of the form convolution-BatchNorm-ReLu. For the related code assembling the generator and discriminator, see the scripts ``mainv3.py'' on the linked Github repository.\\
 
A key design consideration for the generator is based on recognizing that the input and output image (the black-and-white sketch and the colorized image) share similar underlying structures via the edges of characters and objects. The standard approach for an image generator is to employ an encoder-decoder network, but this then requires that information for which I wish to preserve in the input successfully travels through the full length of the net, as well as through the bottleneck. To better account for this low-level information sharing between the input and the output, I add skip connections between layers in the generator thus allowing some information to be shuttled across the net more directly. This is referred to as a ``U-Net'' implementation and each skip connection implies a concatenation of all channels at the two layers. \\

To provide context for the discrimator design choices, I first begin by commenting on the objective function used in this work. Motivated by previous studies in the literature, it is beneficial for this task to use a total loss that is a weighted combination of the GAN objective and the more traditional L1 Euclidean distance. Specifically, while the GAN loss forces the generator to create a result which fools the discriminator, the added L1 loss serves to further anchor the output by enforcing the constraint that the colors in the generated image appear similar to the ground truth image. Furthermore, while there may be many plausible colorization outcomes for a given black-and-white image, the L1 loss can encourage a particular color instantiation over other options especially for a character the network may have had limited exposure to previously. The final objective can be written as,
\begin{equation}
G = \text{arg }\text{min}_G\text{ max}_D \text{ } \mathcal{L}_{cGAN}(G,D) + \lambda \mathcal{L}_{L1}(G),
\end{equation}
where $\lambda$ is the weighting factor and has been set to $\lambda= 100.0$ in this work. A numerical study of different values for the weighting factor was not conducted here and may be a future line of exploration.\\

While the L1 objective (like the L2 distance) is known to cause a blurry output image thus implying a loss of information in transferring high-frequency spatial components, it does, however, still accurately transfers and captures information in the low spatial frequency channels. As a result, by using the joint objective, one then only really require the GAN objective to enforce correctness on high-frequency structure. Because of this fact, one can achieve high quality results with less computational resources by implementing a unique discriminator architecture referred to as a PatchGAN [Pix2Pix paper \cite{Pix2Pix Paper}]. In this approach, the generator image is subsection into patches of pixel size NxN and the discriminator then examines the structure within each patch (now taken to be independent of any long range correlations outside the NxN subsection). Each patch is then classified as real or fake and the discriminator is applied convolutionally across the image. For this work, each logical output from the discriminator here refers to a 70x70 pixel receptive field in its input [PatchGan Refernece \cite{PatchGanGuide}]. Variations to this value were not probed or tested for this study. 


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
