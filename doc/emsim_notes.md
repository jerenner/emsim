# EMSim Notes

## 14 SEPT 2023: Edge data analysis

Here we describe a second attempt to refine edge data with a CNN. 1024 dark-noise-subtracted images of size 3240x2304 were considered. The sum of these images (normalized) is shown below:

![](fig/20230914/summed_img.png)

In summary, in this study we seek to:

1. Quantify the "edge" as a single line
2. Extract hits near the edge in 11x11 subimages as training events
3. Train a neural network to reconstruct events near the pixels with the most charge but avoiding reconstruction on the side with no edge
4. Evaluate the performance of the reconstruction

**1. Quantification of the edge**

To determine the edge, we consider +/- 50 samples in a single row about a center point x0 and fit an s-curve (sigmoid, y = L / [1 + exp(-k*(x-x0))] + b) to the result, similar to the one shown:

![](fig/20230914/edge_fit_single_scurve.png)

The fit x0 is recorded for each row. For all rows for which the fit failed, x0 is considered to be the mean of the values from the neighboring rows. A line is then fit to all of the determined x0 values for each row:

![](fig/20230914/edge_fit_line.png)

The line was found to be y = 6.726618999489237x - 11435.748631593482.

**2. Extraction of 11x11 training samples**

In order to identify individual hits, we must obtain some idea of the noise profile and eliminate noise samples. Looking at a histogram of all samples in a single image,

![](fig/20230914/test_img_samples.png)

we choose a threshold of 9500. Here is a subset of that image with all samples below the threshold set to 0:

![](fig/20230914/test_thresholded_img.png)

We then select rows at random and search for hits near the sample in that row through which the fit line passed. The subimage will be centered about the pixel through which the fit line passed and include +/- 5 samples from that pixel in each dimension (for a total of 11x11 pixels). As later we will present the neural network with images centered about the pixel with maximum charge, the subimage must meet the following conditions:
- at least 1 pixel must be above threshold
- the pixel with maximum charge must be more than 2 pixels from all edges of the subimage

A "centered" subimage (centered about the pixel with maximum charge) will also be produced at this point. The equation of the fit line shifted to the coordinate system of the subimage is also saved. Once an image has been selected, all rows in that image are removed from the array of potential rows from which a new subimage can be originated. Note that for an image to be selected, none of its rows may have appeared in any other selected subimage. Some example subimages are as follows:

![](fig/20230914/subimg_0.png)
![](fig/20230914/subimg_1.png)
![](fig/20230914/subimg_2.png)
![](fig/20230914/subimg_3.png)

Of order 20000 subimages were extracted from the 1024-image dataset to be used for training.

**3. NN training**

A convolutional neural network was then trained to compute the displacement from the center of the pixel with maximum charge corresponding to the electron hit location for each subimage. This was done by presenting the network with a 11x11 subimage centered about the maximum pixel and minimizing a loss consisting of two terms:

1. The 3x3 centroid loss. This term was the distance between the reconstructed point and a the centroid of the 3x3 set of pixels including the maximum pixel. (Another version of the training was also considered in which this term was a "cluster loss" consisting of the minimum distance between the reconstructed point and a "cluster" of pixels with charge greater than some chosen fraction of the maximum pixel charge in the image).
2. The "line" loss. This term is the exponentiated distance from the reconstructed point to the fit line: exp(-0.5*dist_line/sigma_dist), where sigma_dist is a chosen parameter.

An example of two training loss curves is shown below:

![](fig/20230914/loss_1pt0.png)
![](fig/20230914/loss_1pt75.png)

Note that for the "weaker" line loss (sigma_dist = 1.75) the net optimizes more towards a lower centroid term.

**4. Evaluation of the performance**

To evaluate the performance of the network all 1024 images were reconstructed by selecting 11x11 subimages, encompassing pixels above threshold and centered on the maximum pixel in the subimage, using 3 methods:

1. "Threshold" method: assuming the reconstructed point is the center of the pixel with the maximum charge for all subimages.
2. "3x3 centroid" method: assuming the reconstructed point is a 3x3 centroid about the pixel with the maximum charge.
3. NN method: reconstructing using the trained NN.

The reconstructed points were then binned in an image of the original size (3240x2304) with 10 bins per pixel, and the resulting image was rotated so that the edge was vertical, for example:

![](fig/20230914/reco_fac1_10.png)

The rotated image was projected onto the x-axis and the resulting s-curve fit with a sigmoid for each method:

![](fig/20230914/scurves_1pt0.png)
![](fig/20230914/scurves_1pt75.png)

For each of the NNs shown (sigma_dist = 1.0 and sigma_dist = 1.75), the model weights from the final training epoch (200) were used. The conclusion was that it is difficult to distinguish the performance of any of the methods in this way, and no combination of NN training/parameters yielded significant improvement according to these curves.

## 9 JAN 2023: Distribution of distances to the line

In attempting to train the edge net, one might anticipate the goal to be to train a network that improves the sharpness of the line. One way to examine the behavior of the network would be to look at the histogram of distances from the line for many reconstructed points.

For a CNN trained on 400k 11x11 sub-frames (with the revised edge determination discussed in the 12 DEC note below), here are these histograms using the reconstructed strike locations of 100k sub-frames not used in the training performed 1. with the CNN and 2. assuming the reconstructed location is the center of the pixel with the maximum charge:

**Both methods**

![](fig/20230109/distance_val.png)

**Max charge only**

![](fig/20230109/distance_val_maxcharge.png)

**NN only**

![](fig/20230109/distance_val_NN.png)

**Zoomed**

![](fig/20230109/distance_val_near0.png)

A similar exercise can be repeated with MC-generated events (for which we also have access to the true information):

**Max charge**

![](fig/20230109/MC_max_distance.png)

Note that for the MC events the line was always set at exactly the same place, and so the distance to the centers of the neighboring pixels is discrete.

**NN reconstruction**

![](fig/20230109/MC_NN_distance.png)

**True points**

![](fig/20230109/MC_true_distance.png)

**NN and true superimposed**

![](fig/20230109/MC_true_and_NN_distance.png)

## 12 DEC 2022: Refined determination of the "edge"

The "edge" for the sub-images used in training is now determined using a fit to the determined edge points over all rows:

![](fig/20221212/subimg_xmid_215.png)

Here each of the small red dots represents an S-curve fit (in the X-direction) for each row of pixels in the original "edge" image. This fit is done once per row (so the Y-coordinate is the center of the row in Y) but has sub-pixel resolution in X. The line is now determined by a fit to all of these points present in a single 11x11 subpixel. Before we were just drawing a straight line between the points in the top and bottom rows. In the above example, fitting all the points does not seem like it would make much of a difference, but for other sub-images with more dispersion in the points, it would, for example:

![](fig/20221212/subimg_xmid_337.png)

## 22 MAY 2022: Processing of edge data for extraction of single-electron strikes

We begin from 31 data files, each of which contains 4096 frames (512x512), from which 2048 images are constructed by subtracting the first of each set of 2 frames from the other. The median value is subtracted from each image before further processing.

**In short, we want to extract single-electron strikes from this dataset for events near the knife edges.**

First the edge must be determined. This is done as follows:
- read all images from all data files (2048 images per file)
- apply a threshold (100 counts) to determine possible single-electron hit pixels
- sum all counts from each image in each data file to get a summed image for that file
- sum all summed images to get a final image for which the edge will be located
- locate the edge by fitting an s-curve for each row in the final image

Here are the summed images for each dataset:

![](fig/20220522/img_th_all_datasets.png)

And here is the fit edge:

![](fig/20220522/img_edge_fit.png)

The s-curve function used is: L/(1 + exp(-k(x-x0))) + b, and the fit parameter x0 determines where the edge is for a particular row (for now, rounded to the nearest pixel value). Here is an example fit (row 280):

![](fig/20220522/scurve_fit_280.png)

Note that if the fit failed for a row (this occurred for example in the empty row just before 300 in the summed image), the x0 value for the previous row was used.

Next, we must find single-electron strikes near the edge for use in training the network. This is done for each image in each data file as follows:
- the results of the edge fit (the x0 for each row) are stored in an array (of length 512), containing values x0[row]
- a corresponding array of 512 ones is created, corresponding to the edge locations available for selection
- a random edge location (row) is selected from the array, and, if that location is still available (corresponds to a 1 in the array of available edge locations) an 11x11-pixel "event" is constructed, centered upon the corresponding (row,x0) point. If that 11x11 event is considered a valid "subimage" (it contains at least 1 count that falls > 2 pixels from the edges of the 11x11 window), it is kept. The "knife edge" for that image is considered to be the line connecting the points p1 = (row - 5 + 0.5, x0[row-5] + 0.5) and p2 = (row + 5 - 0.5, x0[row+5] + 0.5). The subimage (of the thresholded and non-thresholded image) and line parameters (m and b) are saved. Note that if the line is practically vertical (x0[row-5] == x0[row+5]), it is chosen to slope negative p2 --> (row + 5 - 0.5, x0[row+5] + 0.5 + 1) or positive p2 --> (row + 5 - 0.5, x0[row+5] + 0.5 - 1) to follow the general trend of the edge. In the present example, for the edge with the light region on the left, we see that the fit line is in general sloping negative.
- once a particular edge location (row) has been considered **and leads to the production of a valid subimage**, it is removed from consideration (set to -1 in the array of available edges) along with all elements +/- 5 indices from it in the array
- if an edge location is selected and is available, but not all edges within +/- 5 array elements are available (1's), it is set to -1 along with the elements within +/- 5 indices
- if an edge location is selected and unavailable, the selection continues without modifying the array of available edge locations
- if an edge location is selected and is too close to the first or last row to create an 11x11 event centered on this row, all edge locations within +/- 5 indices of the selected index are set to unavailable (-1's)
- this process continues until all edge locations are set to unavailable (-1)

The resulting set of non-thresholded subimages and their line parameters should serve as a dataset for training the edge network. Here are some examples:

![](fig/20220522/example_strike_0.png)

![](fig/20220522/example_strike_1.png)

![](fig/20220522/example_strike_6.png)

For the left edge of the present example, 587509 training samples were selected in this way.

## 27 SEP 2021: 3x3 single-electron reconstruction, training on NERSC

The basic CNN was modified:
- the event was centered on the maximum pixel chosen from the original event with noise, restricted to an 11x11 square
- a final 11x11 input single-electron event was constructed about the chosen center from the original 101x101 MC event
- a softmax output was used on a 60x60 prediction grid which spanned 3x3 input pixels (a 5 micrometer / 20 = 0.25 micrometer resolution)

**Loss curves**

The accuracy shows whether the pixel in the 60x60 error grid in which the true incident location fell coincided with the maximum of the softmax distribution predicted by the NN on that grid.

![](fig/20210927/training_run_11x11_chi32_60.png)

**An example event**

![](fig/20210927/evt80388_run_11x11_chi32_60.png)

**Errors for 10k events**

Here are the errors in the predicted incident location for all 10k events.

![](fig/20210927/err_run_11x11_chi32_60.png)

In the above, the mean NN error was **0.00408 mm** and the mean 3x3 centroid error was **0.00648 mm**.

If we consider *only events reconstructed to an error < 0.005 mm* (7115 events for the 3x3 method, 7459 events for the NN method):

![](fig/20210927/err_zoom_run_11x11_chi32_60.png)

In the above, the mean NN error was **0.00112 mm** and the mean 3x3 centroid error was **0.00115 mm**. Note that the Gaussian fit to the softmax distribution providing the mean reconstructed location is failing ~2-4% of the time, in which case the reconstructed location corresponds to the center of the pixel with the maximum probability in the softmax distribution.

## 17 SEP 2021: single-electron reconstruction problem, training on NERSC

Here we trained a basic CNN, on 21x21 events for which the central pixel was known to be the correct pixel, to reconstruct the exact incident electron location. The idea was to compare the NN results to a 3x3 centroid method. The CNN reconstructed a softmax distribution over a 10x10 error grid covering the central pixel. The distribution is then fit to a 2D Gaussian, and the mean of that Gaussian is taken to be the predicted location.

**Loss curves**

The accuracy shows whether the pixel in the 10x10 error grid in which the true incident location fell coincided with the maximum of the softmax distribution predicted by the NN on that grid.

![](fig/20210917/training.png)

**Errors for 10k events**

Here are the errors in the predicted incident location for all 10k events.

![](fig/20210917/errors_all.png)

In the above, the mean NN error was **0.00152 mm** and the mean 3x3 centroid error was **0.0629 mm**.

If we consider *only events for which the 3x3 centroid reconstructed to an error < 0.005 mm* (653 events):

![](fig/20210917/errors_3x3_lt_5microns.png)

In the above, the mean NN error was **0.00150 mm** and the mean 3x3 centroid error was **0.00188 mm**.

So it looks like overall the NN performs significantly better, however there are a few events that were better-reconstructed by the 3x3 centroid method. Zooming in on the above (3x3 error < 0.005 mm) histogram to the < 0.005 mm region (653 events in the 3x3 centroid error histogram, 636 events in the NN error histogram):

![](fig/20210917/errors_3x3_lt_5microns_zoom.png)


## 17 SEP 2021: s-curve comparisons

Here we compare the s-curves over 100k single-electron MC events for the following cases. Note that because the NN methods are not forced to count, the total number of events counted may be less than 100k. In the construction of the s-curve, we normalize so that the maximum count value in the summed image of all counts is equal to 1.

- **classical** (100k counts): counting done by the classical method of choosing the maximum pixel value
- **UNet, no edge, epoch 500** (NN threshold = 0.88, yielding 85534 counts): UNet was trained to learn the classical algorithm with no edge information
- **UNet + edge, epoch 11** (NN threshold = 0.8, yielding 85135 counts): UNet was trained to learn the classical algorithm with the edge restriction, but we are only considering training up to epoch 11
- **UNet + edge, epoch 500** (NN threshold = 0.8, yielding 84882 counts): UNet was trained to learn the classical algorithm with the edge restrictions for 500 epochs
- **true** (100k counts): the true counts

**Full plot**

![](fig/20210917/scurve_all.png)

**Zoom, light side**

![](fig/20210917/scurve_all_zoom_upper.png)

**Zoom, dark side**

![](fig/20210917/scurve_all_zoom_lower.png)

## 11 SEP 2021: another attempt at using edge information

Changing the strategy a bit:

- again using only single-electron events (all events have exactly one count in the light region) on a 20x20 grid
- the truth was constructed by a "classical" algorithm, taking the maximum pixel as the electron count. However, before determining the maximum pixel, the "edge" was applied: all sensor input in the dark region was set to 0. Therefore, the "true" count could never be determined
- the usual binary cross-entropy loss was used. Therefore the effects of the edge were completely accounted for in the construction of the truth and not in the calculation of the loss.

Below are some key results. Note that the UNet threshold chosen in each case was meant to give about the same true positive rate as the classical algorithm (1 count at the simple maximum). It remains to be studied whether or not this was the best choice.

**Training**

![](fig/20210911/training_edge_argmax_noweights.png)

**Edge fits**

![](fig/20210911/edge_fit_noweights.png)

**Edge plot**

![](fig/20210911/edge_plot_noweights.png)

Note that performing the same training without incorporating the edge information into the truth (also training for 500 epochs) does seem to give a result similar to the classical method (no improvement).

**Edge fits (no edge used in truth)**

![](fig/20210911/noedge_fit_noweights.png)

**Edge plot (no edge used in truth)**

![](fig/20210911/noedge_plot.png)

Note: Event weighting (favoring events near the edge) seemed to bias reconstruction of the events towards the edge.

**Edge fits with event weighting**

![](fig/20210911/edge_fit.png)

**Edge plot with event weighting**

![](fig/20210911/edge_plot.png)

## 07 SEP 2021: further attempts to train on edge events

Here we consider another scenario in which no assumption is made about how to reconstruct a single electron, but the information about the edge is used to penalize reconstruction in the dark region.

- only single-electron events were generated. That is, all events were guaranteed to have exactly one count in the light region.
- the loss contained:
  - the penalty term (equal to the absolute distance from the line multipled by the predictred pixel value, summed over all pixels in the dark region)
  - a term equal to |(the sum of all predicted pixel values) - 1|, essentially attempting to restrict the output distribution to a "single count"
- events were weighted with a Gaussian (sigma = 1 pixel) in the distance from the line

The training loss (again accuracy is computed with respect to the *correct* truth):

![](fig/20210907/training_edge.png)

In the end the resulting reconstructed events did not make sense and favored pixels near the line:

![](fig/20210907/evt_example0.png)

![](fig/20210907/evt_example1.png)

![](fig/20210907/evt_example2.png)

## 04 SEP 2021: attempts to train on edge events

**Adaptive truth**

Here we attempt the following:
- assume a single-threshold (= 825) truth
- use a loss including the usual binary cross-entropy + a penalty equal to the absolute distance from the line multiplied by the pixel value for all pixels in the dark region
- weight events with a Gaussian (sigma = 1 pixel) in the distance from the line
- slowly adapt the truth to be more and more like the output of the net (`truth = (1-f)*original_truth + f*output`, where `f = epoch/500` for 500 training epochs)

The loss and accuracy (note accuracy is computed with respect to the *correct* truth):

![](fig/20210904/training_adaptive_truth.png)

The ROC curve (note NN thresholds between 0.0001 and 0.9995 are covered):

![](fig/20210904/ROC_adaptive_truth.png)

The line fits:

![](fig/20210904/lines_adaptive_truth.png)

The edge plot:

![](fig/20210904/edge_adaptive_truth.png)

## 28 AUG 2021: possible strategy for data-based training

We have been considering how to use "edge" data to train a neural network, either to improve an already-trained (on MC) network or to

One potential solution was to change the loss layer of an MC-trained network, and perform a "refinement" step with a new loss that penalized counting within the "dark" region and allowed counting in the "light region". The problem with this is that the loss must also consider what is actually being counted: we cannot tell the net to only count in one region without telling the net how to count. Such a loss would result in an optimal solution of a "blank" frame - if in one region no counting should be performed, and in another region it doesn't matter, but there is no reward for actually specifying a count anywhere, an easy zero-loss solution would be to not count at all.

So now we consider the following strategy: count using some "classical" algorithm and teach the net to do the same, however prohibit counting in the dark region. Therefore we use the same UNet with the same (binary cross-entropy) loss we had been using previously, but construct the "labels" as the result of some classical counting algorithm, for example the single-threshold. This can all be done directly on data, and we can add the additional information provided by the edge by performing an AND of the classical label with the "light" edge, as follows:

![](fig/20210828/combined_truth.png)

This event had 2 electrons thrown on the light side of the edge, however it looks like at least one event produced significant activity on the dark side. The classical algorithm finds electrons on both sides of the edge, but as we know that no count on the dark side should be possible, we can modify the truth to reflect this. Let's look at what effect this has on the net.

**UNet trained on the classical threshold**

First we just try to train UNet to reproduce the classical algorithm. Training for 200 epochs to a loss of about 6e-4 gives the following ROC curve:

![](fig/20210828/ROC_classical.png)

Note that the leftmost band of UNet points (true positive rate about 0.6 to 0.8) spans NN thresholds of about 0.997700 to about 0.999995, so the network does not "naturally" operate in this regime. Its performance is not strikingly different compared to that of the actual algorithm in the regime of true positive rate 0.84 - 0.89 which corresponds to the majority of NN threshold values (0.03 - 0.97).

Now generating 20x20 edge events and performing the fits:

![](fig/20210828/edge_fits_classical.png)

Note we used m = -2.0, b = 30.0 in the generation of the events. This gives rise to the edge plot:

![](fig/20210828/edge_plot_classical.png)

**UNet trained on the classical threshold + edge info**

Now we repeat the procedure, but ANDing the edge truth with the classical threshold truth, and using the result to train the network. After 200 epochs, we have a loss of about 1.9e-3, and the following ROC curve:

![](fig/20210828/ROC_edge.png)

And edge fits (for NN threshold of 0.6 this time, corresponding to about an 85% true positive rate):

![](fig/20210828/edge_fits_edge.png)

And edge plot:

![](fig/20210828/edge_plot_edge.png)

It's possible that we're actually getting some improvement by adding the additional edge information, though it's uncertain whether this is due to actual use of this information or differences in the training procedure itself.

## 25 AUG 2021: MC edge events

To further develop the ideas behind fitting and interpreting the edges, MC events prepared to match data (as described in the notes from 15 AUG 2021) were generated, though with the condition that all electrons for which the incidence pixel fell under some specified line were not placed.

- 10000 such frames of size 50x50 were generated, and each one was counted using a classical threshold and UNet. The thresholds (0.14 for UNet, and 825 for the classical threshold) were chosen to admit approx. 80% of electrons in each case.
- The 10000 counted frames were summed in 4 cases:
  1. before counting (the raw pixel values themselves were summed)
  2. true counts
  3. classical threshold counts
  4. UNet counts
- Each summed 50x50 array was then "normalized" by dividing each pixel by the maximum value.
- The edge was determined for each normalized array in a procedure similar to that described in the notes from 20 AUG 2021, with some differences:
  - as for 10000 frames, the "occupancy" of the summed frame was high (few blank spaces on the side of the line containing counts), no pre-determined weights were appled to L2 and L3 (w = 1)
  - each value determined to be "0" (less than some specified threshold) was counted in the appropriate sums (L1 and L4) as a value equal to 1 - [the pixel value at that point], and each value determined to be "1" (greater than some specified threshold) was counted in the appropriate sums (L2 and L3) as a value equal to [the pixel value at that point]

Using an input line with m = -2.0 and b = 80.0, the edges determined for the summed frames, the true counts, the UNet counts, and the classical counts are shown below along with the noise thresholds used and parameters (m and b) determined by the minimization procedure in each case:

![](fig/20210825/edge_fits.png)

Note that thresholds of 0.35 were used in the case of the counting methods. Because the raw frame pixels were not noise-suppressed with some counting method, all of the pixel values were "near" the maximum value, and a specific threshold of 0.988 was chosen by eye. For a reasonable threshold, it does not appear to be too difficult to determine the correct line parameters.

Looking at a plot of the average pixel value vs. distance from the fit line in each case:

![](fig/20210825/line_sharpness.png)

Here one can see the two regions with a step-function-like decrease in average pixel value along the line. (One can see that 0.35 was approximately 1/2 the average value of the "light side" pixel value in the true and UNet cases, and therefore this value was chosen as the noise threshold in the line fit procedure.) In principle these curves could be fit to determine more quantitatively the blurriness of the lines. By eye, one can see that the line gets progressively blurred when going from true counting, to UNet-based counting, and finally to classical threshold-based counting, and this is evident in both the step-function plot and the distributions showing the fit lines above.

From this one can conclude:
- while in the best case, we will know the line equation for real data, it is possible to reasonably fit a line with the proper choice of background threshold (it seems reasonable to use 1/2 of the "step" value; one may have to perform a rough fit first to determine what this value should be)
- given a sensible line fit, the quality of the reconstruction could most likely be quantified by examining the profile of mean pixel value vs. distance from the line

Now the question is: what would the step-function curve look like for real data when counted with UNet, and would it be better than the curve above for counting MC events?

## 20 AUG 2021: idea for fitting the data "edge"

One potential way we can determine the line and perhaps measure the "fuzziness" of the edge for data taken with distinct light/dark regions is by finding the line that maximizes a loss function consisting of 4 components:

- L1 (+ contribution): the number of 0's in the dark region
- L2 (- contribution): the number of 1's in the dark region
- L3 (+ contribution): the number of 1's in the light region
- L4 (- contribution): the number of 0's in the light region

The loss is then:

L = L1 - w\*L2 + w\*L3 - L4

where w is a weight factor equal to (the number of 0's in the image) / (the number of 1's in the image) and serves to give 1's and 0's an equal "weight" in the loss. Without this weight the maximal loss may occur by simply placing the line such that the entire image is in the dark region (the 0's dominate).

Assuming the x-axis (columns) increases to the right and the y-axis (rows) increases downwards, the dark region lies above the line in the following plots and the light region lies below the line (r = m*c + b, for row r, column c, and parameters m and b). We set all pixels below some noise threshold to 0 and consider all others to be a count (1), and starting from an initial guess for the line:

![](fig/20210820/edge_initial_guess.png)

a minimization of -L using `scipy.optimize.minimize` with the 'Nelder-Mead' method gives:

![](fig/20210820/edge_fit.png)

(m = -7.11, b = 26980.75)

From here one could possibly measure the "fuzziness" of the line by plotting the relative loss obtained varying one or both parameters near the solution. For example, for parameter m:

![](fig/20210820/m_vs_L.png)

A well-determined line will likely show steeper variation in the loss over the same range near the solution, though this will have to be confirmed.

## 18 AUG 2021: initial evaluation of UNet on data
*Correction made 20 AUG 2021: The 5760x4092 image should have in fact been read as 4092x5760, and this caused strange effects in the data which have since been corrected in the example images shown below.*

The results of an initial evaluation of real data with the UNet trained on data-matched MC seems to give sensible results.
- Initially, it was attempted to run the entire 4092x5760 image through the UNet (trained on smaller 50x50 images) at once.
- The full image was too large to process on an 8 GB GPU (ran out of memory), as was a 2046x2880 quarter-image.
- However, an eighth-image 1023x1440 was able to run and produce a corresponding 1023x1440 output (see examples below).
- So far, the potential effects of training on smaller images and evaluating a larger image have not been studied.

Here are some examples of two 50x50 subsets of an evaluated 1023x1440 eighth-image:

![](fig/20210818/data_eval_example1.png)

![](fig/20210818/data_eval_example2.png)


## 15 AUG 2021: training UNet on data-like MC

Reading the first 5670x4092 unsigned integers from the data file "stack 1.dat" gives the following distribution of pixel counts:

![](fig/20210815/counts_data_stack_1.png)

Making the same plot using a large MC-constructed event with a similar number of pixels (4855x4855), an electron occupancy similar to the frames from the smaller 4dstem dataset (22 electrons for a 50x50 region), and a noise sigma of 20:

![](fig/20210815/counts_MC.png)

Zooming in on the area near the noise:

**Data distribution, zoomed to the lowest noise values**

![](fig/20210815/counts_data_stack_1_zoom.png)

**MC distribution, zoomed to the lowest noise values**

![](fig/20210815/counts_MC_zoom.png)

The idea is now to attempt to make the MC events more similar to the real data. Fitting the noise peak in data:

![](fig/20210815/fit_noise_peak_data.png)

Now using a noise distribution with mean 683 and sigma 11.2 and applying a scale factor of 1/12 to the data distribution gives:

![](fig/20210815/data_and_MC_distributions.png)

Training UNet on MC (50x50 events) with the above noise distribution and making the ROC curve of true positive vs. false positive gives for the net and a constant threshold:

![](fig/20210815/true_positive_vs_false_positive.png)

The idea now would be to use this net to count real data. Is it possible to apply a net trained on 50x50 frames to a much larger frame?

## 2 AUG 2021: electron counting ROC curve for Unet vs. classical threshold approach

Here is the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) for the electron counting averaged over 100 frames:
- with Unet, thresholds [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
- a basic threshold approach in which all pixels above a given threshold were considered "counts", for thresholds [  0,  20,  40,  60,  80, 100, 120, 140, 160, 180]

The frames were generated in the same configuration as discussed in the note on 28 JUL 2021 (50x50 frames with 22 +/- 0.5 electrons per frame and a noise
sigma of 20 counts).

![](fig/20210802/roc_curve_ec.png)

Zooming in on the relevant region:

![](fig/20210802/roc_curve_ec_zoom.png)

Some notes:
- In order to be considered a true positive, the counted pixel must coincide exactly with the corresponding true pixel. In many cases a count may be predicted in the correct general area of a true electron, but may be assigned to a neighboring pixel, in which case it would not be considered a true positive.
- Even with relatively low confidence thresholds (0.05) Unet does not seem to get beyond the 60-70% true positive range
- A false positive rate of 0.0088 gives a number of false positives equal to the mean of the number of true electron counts (22.0)


## 28 JUL 2021: electron counting with UNet

An initial training of UNet ([https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)) has been performed for counting electrons using
the MC frames. No "validation set" was used, as the idea is that no 2 events
will be exactly the same, however it does in fact make sense to use a
different set of MC single-electron events in constructing the frames for a
validation set, so this can be done in the future. For this initial test,
50x50 frames were generated with 22 +/- 0.5 electrons per frame and a noise
sigma of 20 counts.

UNet assigns a value of 0-1 for each pixel, corresponding to the confidence
that that pixel was the entry point of an electron. The threshold on the confidence
can be varied to accept more events at the expense of more potential false counts.

A few initial examples (note changing the threshold only affects the final count
shown in the title of the plot on the far right):

**Threshold 0.9**
![](fig/20210728/EM_NN_UNet_50x50_0pt9_0.png)
![](fig/20210728/EM_NN_UNet_50x50_0pt9_1.png)
![](fig/20210728/EM_NN_UNet_50x50_0pt9_2.png)

**Threshold 0.1**
![](fig/20210728/EM_NN_UNet_50x50_0pt1_0.png)
![](fig/20210728/EM_NN_UNet_50x50_0pt1_1.png)
![](fig/20210728/EM_NN_UNet_50x50_0pt1_2.png)


## 26 JUL 2021: frame generation

A random frame containing many electron events can be generated by:
- initializing the frame as a 2D array of all 0's
- casting a random number for the number of electrons to be generated Ne
- for each of Ne electrons:
    - casting a random number corresponding to the central pixel of the electron
    - choosing an event at random from a dataset of single-electron events
    - adding the pixel values of the randomly chosen single-electron event to the frame, centered upon the chosen pixel
- add a specified amount of noise to the entire frame

Here are several events constructed in this way without noise, for a frame size of 100x100 and 10 +/- 1 electrons per frame:

![](fig/20210726/frame_no_noise_0.png)

![](fig/20210726/frame_no_noise_1.png)

And with Gaussian noise (mu = 0, sigma = 20):

![](fig/20210726/frame_with_noise_0.png)

![](fig/20210726/frame_with_noise_1.png)

Here is a 576x576 frame with many (2927.294 +/- 70.531 used in generating the random number), shown on a log scale.

![](fig/20210726/frame_576.png)


## 24 JUL 2021

Training a net with a 10x10 prediction grid and no displacement of the
central pixel seems to give better results for the neural net. Here we trained
with learning rate 1e-3 for 73 epochs and then 1e-4 for the rest:

![](fig/20210724/EM_NN_training_10grid.png)

Here are the errors in the determined positions (note that in the case of the 3x3
centroid method, the pixel of electron incidence was still assumed to be the one with the most charge, and not the central pixel):

![](fig/20210724/EM_NN_errors_NN_vs_3x3.png)

![](fig/20210724/EM_NN_errors_NN_vs_3x3_zoom.png)

Here it looks like the NN gives similar performance for well-identified events. In fact, some of these events in the longer shoulder of the NN
distribution must actually correspond to events with much larger error using the 3x3 method. Looking at the error differences,

![](fig/20210724/EM_NN_error_differences_NN_vs_3x3.png)

perhaps there is still a slight asymmetry in favor of the 3x3 centroid for
low error differences, but the NN still seems to give similar performance.

We note that the above comparison was somewhat unfair - the NN was essentially
restricted to predicting within the central pixel (some predictions fell outside
of this because the incident location was taken to be the mean of the 2D Gaussian,
which may have been outside the 10x10 grid for events near the edge) while the
3x3 method was not told that the central pixel was correct. If we now:

- demand for the 3x3 method that the central pixel is always correctly chosen
- use the point of maximum predicted probability within the 10x10 grid for the NN (not the mean of the Gaussian fit)

Here are the errors:

![](fig/20210724/EM_NN_errors_NN_vs_3x3_centralpixel.png)

`Mean 3x3 error -- 0.0011584349887750222`<br>
`Mean NN error --- 0.0011583734270575604`

and the error differences

![](fig/20210724/EM_NN_error_differences_NN_vs_3x3_centralpixel.png)

## 23 JUL 2021

The net was trained further with a prediction grid of 126x126 with learning rates of 1e-3, 1e-4 (first jump in loss) and 1e-5 (2nd jump in loss):

![](fig/20210723/EM_NN_training_126grid_continued.png)

It looks like most additional gains were made with the 1e-4 learning rate. Looking
at an individual event (using the model after the training with the 1e-4 learning rate):

![](fig/20210723/EM_NN_evt81000_126grid_1.png)

It looks like the uncertainty of the net prediction is still high, and we can also
see the structure of the 21x21 event pixels on the 126x126 prediction grid.

## 22 JUL 2021

The net should be able to match the standard method at low error. Maybe there is a way to weight the loss toward easier clusters, but I would think that already the net is getting more exposure to easier events as there are more of them. Looking at the NN error minus the 3x3 (0 threshold) error, it looks like when the error difference is less than 2 pixels, the NN is more likely to give greater error (the difference is positive):

![](fig/20210722/EM_NN_error_vs_3x3_error_zoom.png)

However for large error differences, the NN "wins":

![](fig/20210722/EM_NN_error_vs_3x3_error.png)

I guess this is just another way of looking at what we already understood from the previous plots. Now with the NN "knob" at the ~80% efficiency level:

![](fig/20210722/EM_NN_error_vs_3x3_error_NN_knob_eff80_zoom.png)

![](fig/20210722/EM_NN_error_vs_3x3_error_NN_knob_eff80.png)

So the knob seems to help the NN much more than the standard method, but still at pixel-level error differences the standard method seems to win by a bit.

I tried to use the Gaussian mean rather than the highest probability error pixel, and it didn't seem to help. Maybe a finer grid would help - perhaps I could try 126x126, dividing each pixel into 6x6

I've done an initial training on a 126x126 prediction grid, so far just the round with the higher learning rate (1e-3):

![](fig/20210722/EM_NN_training_126grid.png)

It looks like the predicted region gets cloudier:

![](fig/20210722/EM_NN_evt81000_126grid.png)

I'll keep going with learning rate 1e-4, but I think the grid may be getting too fine (maybe I need to increase the size of the net to get enough parameters to get a good fit). I may want to run these tests with no displacement of the central/max pixel and over a 10x10 grid (covering the 1 pixel we know for sure is the one corresponding to electron incidence), just to make sure that in the simplest case the NN can match the performance of the basic centroid method. Then we can expand to the wider field-of-view with the +/- 10 pixel shift.

## 21 JUL 2021

Here is a "scatter matrix" of the errors for 10000 validation events, noise sigma = 20:
- error_r_NN: the error on the NN-predicted quantity (pixel with max prediction on the 84x84 grid spanning 21x21 pixels)
- error_r_maxpt: the error when the center of the pixel (in the 41x41 event) with maximum counts is used as the electron location
- error_r_3x3: the error when the centroid of the 3x3 region around the maximum pixel (in the 41x41 event) is used
- error_r_3x3_th: same as error_r_3x3 except only using pixels above a threshold of 40 counts
- error_r_5x5:the error when the centroid of the 5x5 region around the maximum pixel (in the 41x41 event) is used
- error_r_5x5_th: same as error_r_5x5 except only using pixels above a threshold of 40 counts

![](fig/20210721/EM_NN_errors_scatter_matrix.png)

Here the non-NN errors are more correlated, and it looks like the number of events with (low NN error) and (high error in the non-NN quantities) is greater than vice versa, that is, that the NN seems to be "way off" on fewer events. However, looking more closely at the region of low error (here we compare the NN vs. the 3x3 thresholded centroid):

![](fig/20210721/EM_NN_vs_3x3_thresholded.png)

It looks like the net actually does worse overall, for the events with low error, than the 3x3 thresholded method (probably those for which we get the right pixel by choosing the one with maximum counts). Perhaps the NN could be improved by:
- increasing the resolution of the prediction grid
- using the mean of the Gaussian fit as the predicted location rather than the center of the maximum pixel in the prediction grid

Either way it's still hard to tell at this point how much the NN will actually gain for us, if anything. It does include the "knob", but perhaps we could compute a different non-NN "knob" factor and gain a similar amount.

Here are the scatter plots zoomed < 0.01:

![](fig/20210721/EM_NN_errors_scatter_matrix_zoomed.png)

## 19 JUL 2021

I've continued the training for a few hours after the epoch where it was previously left off (epoch 244). There seems to be a sharp drop in accuracy and increase in loss at that point - I thought I continued with the same learning rate as previously, so I am surprised by this break. It seems to recover slowly:

![](fig/20210719/EM_NN_training_continued.png)

During the initial training I also stopped it and decreased the learning rate a few times to try to improve the results. I started with 1e-3 and then changed to 1e-4. I thought I also ran with 1e-5 for some epochs, though I may not have (that would explain the sudden change in this most recent training run). I could try training again from the beginning, perhaps staying at 1e-3 for longer.

## 17 JUL 2021

I've implemented the gaussian-based sigma determination:
sigma_x and sigma_y are now determined by fitting to: A\*np.exp(-0.5\*((X-x0)^2/(varx)+(Y-y0)^2/(vary))) + C: the initial values are determined by
- the maximum pixel value (A)
- the maximum pixel location (x0,y0)
- the squared sigmas computed over the entire grid, excluding pixels less than a threshold equal to 1/10 of the max pixel value (varx,vary)
- 1/10 of the max pixel value (C)

The final sigma is the squared sum of the sigmas from the Gaussian fit in x and y.

**Here is an example event for several different noise generations, now showing the Gaussian function constructed from the fit parameters**

![](fig/20210717/EM_NN_evt81000_0.png)

![](fig/20210717/EM_NN_evt81000_1.png)

![](fig/20210717/EM_NN_evt81000_2.png)

![](fig/20210717/EM_NN_evt81000_3.png)

**The mean errors and sigmas for 10k validation events**

![](fig/20210717/EM_NN_errors_and_sigmas.png)

**The mean errors and sigmas for sigma < 0.011 (~80% efficiency)**

![](fig/20210717/EM_NN_errors_and_sigmas_cut.png)

**The efficiency curve**

![](fig/20210717/EM_NN_eff_vs_err.png)

## 16 JUL 2021
Here is the summary of the current progress:

**The network training loss and accuracy** (80k training events, 20k validation); not sure why validation set seems to significantly outperform (due to dropout in training?)

![](fig/20210716/EM_NN_training.png)

**An example event from the validation set**

![](fig/20210716/EM_NN_evt81000.png)

**For 5k of the validation events, the errors on the predicted (x,y) values and the calculated sigmas ||<x^2> - <x\>^2, <y^2> - <y\>^2)|| of the predicted probability distributions**

![](fig/20210716/EM_NN_errors_and_sigmas.png)

**For the same 5k validation events, the mean error given a cut on the sigma < sigma_cut, with sigma_cut varied between 0.035 and 0.045**

![](fig/20210716/EM_NN_eff_vs_err.png)

**To illustrate the effect of the "knob" for a specific point on the curve above:**
same distributions of errors on the predicted (x,y) values and the calculated sigmas of the predicted probability distributions, but now with a cut of sigma < 0.041 (gives ~68.5% efficiency)

![](fig/20210716/EM_NN_errors_and_sigmas_cut.png)

Note: the efficiency vs. error curve is only shown above for a range of sigma cuts between 0.035 and 0.045 because strange things happen at lower sigma: here is the plot for 0.025 to 0.045 (not connecting the dots); the efficiencies at these lower sigmas are probably too low to be of interest anyway

![](fig/20210716/EM_NN_eff_vs_err_all.png)

## 15 JUL 2021

I've managed to add to the EM network training:
- shifting of the events by +/- 10 pixels in either direction. As the original 101x101 events were reduced to 21x21 and a 10-pixel shift from the center would start cutting off the patterns, I've expanded the events an additional 10 pixels in each direction, per dimension, so the events are now 41x41 when input to the net
- training of the network to cover a 21x21 pixel range, so the final grid spans 20*0.005 + 0.0025 = 0.1025 in each dimension

So far, I've tried several different grid sizes, most recently 80x80 (perhaps I should use 84x84 so an integer number of grid cells corresponds to 1 pixel). Here is the same event with different shifts on the 80x80 prediction grid:

![](fig/20210715/EM_NN_evt9801_0.png)

![](fig/20210715/EM_NN_evt9801_1.png)

It looks like the strategy is more or less working. Right now we're dividing each pixel into roughly 4x4. How fine do we want to make the grid? Do we need to be more precise than ~1.25 micrometers in each dimension?

I will now see how this works with 20 e- of noise

Here is the first attempt to train with noise (same event as in previous email, now with noise and different shifts). This is an event from the validation set, with several different shifts + noise; the images on the right are the predictions and true locations (red dot):

![](fig/20210715/EM_NN_evt9801_noise_0.png)

![](fig/20210715/EM_NN_evt9801_noise_1.png)

![](fig/20210715/EM_NN_evt9801_noise_2.png)

It trained on 8k events for ~900 epochs. I think it can learn more, especially with more events, though I'm not sure how much more. I will try:
- a 100k event set (80k training, 20k validation)
- similar grid sizes (41x41 events, 84x84 prediction grid)
- looking at prediction error vs. some quantity describing the sharpness of the prediction distribution, perhaps sqrt(sigma_x^2 + sigma_y^2)

## 13 JUL 2021

Here are the errors in the predictions (since the prediction is constrained to be one of 100 points on the 10x10 grid while the incident location could be anywhere within the grid, there is almost always some error)

![](fig/20210713/EM_NN_prediction_error_validation.png)

Here are a few events that had very high error (> 0.002 mm in the x-position):

![](fig/20210713/EM_NN_evt8328.png)

![](fig/20210713/EM_NN_evt9186.png)

For these it looks like something just went wrong in the network because the error distributions still seem sharp (maybe needs more training with more events). Now looking at events with some error but less dramatic (> 0.001 mm in the x-position):

![](fig/20210713/EM_NN_evt8241.png)

![](fig/20210713/EM_NN_evt9155.png)

![](fig/20210713/EM_NN_evt9210.png)

![](fig/20210713/EM_NN_evt9756.png)

## 11 JUL 2021

An update on progress on the network: I've managed to train a net that seems to be giving reasonable initial results:
- starting from the events centered on the pixel in which the electron was incident, with no noise added; I've reduced the size to the central 20x20 region for these initial tests
- dividing the central pixel into a 10x10 grid on which the precise incident location is predicted (so each bin corresponds to 0.5 micrometers of space - we could go to higher resolution if needed)
- training the network to produce a probability distribution over the 10x10 grid; the "label" is the grid filled with a 1 in the bin in which the incident location falls, and 0s everywhere else
- 8k events training set, 2k validation

While the training "accuracy" (whether or not the highest-probability predicted bin corresponds to the true bin) seems to peak at ~40% for a convolutional network, it looks like most predictions are "correct" to within about 1 bin. Here are a few examples from the validation set - using the 4 micrometer, 300 keV data cropped to 20x20 about the central region - left plot is the event, right plot is the distribution over the 10x10 grid with the true bin marked with a red dot. Note the red dot marks the center of the true bin, not the absolute location, which would not always be a bin center:

![](fig/20210711/EM_NN_evt9802.png)

![](fig/20210711/EM_NN_evt9803.png)

![](fig/20210711/EM_NN_evt9804.png)

Some thoughts on this:
- it will be interesting to see how things look once some noise is added. I think that for now we could keep the correct central bin and just add noise, so identifying the correct central bin we assume "solved" for the moment. With this we may start to see events that perform poorly, and perhaps the probability distribution will be less peaked, reflecting the net's confusion with these events.
- we could go back to 100x100 events if necessary - I had cropped to 20x20 to remove the non-central region which seems to be all 0s (though won't be once we introduce noise). Is there a reason we should use 100x100 events rather than cropping them nearer to the central region?

## 09 JUL 2021

I'm attempting to select the amount of noise to include in the electron EM-ML study.

After applying the noise, the new "center" of the event is now determined by finding maxima over 3x3 regions, starting with the 3x3 region containing the maximum pixel, and then trying the 3x3 region surrounding the 2nd-highest pixel. If the 3x3 region surrounding the 2nd-highest pixel actually had a higher sum, we check the 3rd highest, and so on until the next-highest gives a lower 3x3 region sum, in which case we stick with the maximum pixel we are on with the highest 3x3 region sum.

Using this strategy, here is the average error sqrt(xerr^2 + yerr^2) vs. the sigma of the noise in electrons (using 1000, 300 keV events - error bars are the error on the mean = r_sigma/sqrt(1000)):

![](fig/20210709/rerror_vs_noise.png)

The curve seems to be asymptoting at a value equal to about half the distance from the center to one corner of the square (0.5 mm side length), which would be sqrt(2)\*(100 pixels / 2)\*(5 micrometers) / 2 = 0.177 mm. I suppose this corresponds to the point at which the maximum pixel is essentially chosen randomly because the noise dominates most events (though I would have expected this value to lie between 0.177 mm and 0.177 mm / sqrt(2) = 0.125 mm, as the corner is the farthest-reaching point of the square).

I'm not sure where on this curve we should be operating, though for now I will choose something like sigma_noise = 20 electrons.
