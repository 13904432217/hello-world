http://www.futilitycloset.com/2015/10/19/more-fun/

Q: If tall people are more likely to get cancer, then are people who get cancer more likely to be tall?

### Formalize

    C = Cancer
    T = Tall
    Conjecture: P(C|T) > P(C) ⇒ P(T|C) > P(T)

According to Bayes' Rule

    P(C|T) = P(T|C) P(C) / P(T)

so,

    P(C|T) > P(C)
    P(T|C) P(C) / P(T) > P(C)
    P(T|C) > P(T)

yes.


CS401 Machine Learning & Neural Networks
=======================
 
Three main types of Machine Learning
-------------------------------

* Supervised Learning (regression)
* Unsupervised Learning (figure out structure, data mining)
* Reinforcement Learning

Wellsprings of Machine Learning
-------------------------------

* Making computers do things (automatic programming)
* Statistics (making inferences from data)
* Neuroscience and psychology and cognitive science
* Information theory
* Physics (especially thermodynamics)

Think of a baby, that learns and takes in data constantly

In the beginning
----------------

In the dawn of the computer age, computers had poor languages, severe limitations, and high hopes and ambitions. An example problem was translating Russian to English or vice versa. People sat down with Russian-English dictionaries and attempted to write a program that would do it. Another example would be telling a computer to identify a tank in a photograph, or even translate a scanned page of text from a bitmap to text. People are good at these jobs, and computers are not. 

An example to think of would be a big box, with lots and lots of knobs on it. These knobs don’t have set positions; they rotate freely. If people were to spend years trying to get these knobs in the correct position, such that they do something we want, they may never get it to work. Machine learning would enable us to do it automatically. 

Supervised Learning
===================

This is supervised learning, or regression: the task of inferring a function from labelled training data.

Example: Tank recognition in an image

data
----

input   | desired output
-----   | --------------
image 1 | yes
...     | ...
image M | no

Image → [Machine] → Classification, class label, or probability

How does regression work? 
-------------------------

If we call each image a vector (that is, 3 2x2 RGB arrays), then

inputs → desired outputs

x(i) → y(i)

We can call all the settings of all the knobs on our theoretical machine w, with w being a vector (or data structure in code).

x → Machine → ŷ

ŷ = f(x;w)

E = (1/m) sum_i=0..(m-1) (1/2) || f(x(i);w) - y(i) ||^2

Linear Regression
-----------------

f(x;w) = W x

This is linear regression, a special case of regression analysis, which tries to explain the relationship between a dependent variable and one or more explanatory variables.

Examples of machine learning in everyday applications include

* predicting housing prices
* language translation
* face detection in cameras
* voice recognition

Unsupervised Learning
=====================

Another form of machine learning is unsupervised learning, the objective being to try to find hidden structure in unlabelled data. Since the examples given to the machine are unlabelled, there is no error or reward signal to evaluate a potential solution. 

Clouds (i.e., Amazon, Google, etc.) devote a lot more cycles to unsupervised learning than to other forms of machine learning, as a lot of unlabelled data is freely available to them, via an internal database, via data collected by a spider trawling the web, or even via shopping trends. 

Reinforcement Learning
======================

The third, and final type of machine learning is reinforcement learning. This is the most “biological” of the types, inspired by behaviourist psychology, concerned with how software agents ought to take actions in an environment so as to maximize some notion of cumulative reward. That is to say, the machine is put into an environment and is given “rewards” for getting things right. 

Reinforcement learning subtypes:
--------------------------------

* Without time (“bandit” problem -  the problem a gambler faces at a row of slot machines ("one-armed bandits"), when deciding which machines to play, how many times to play each machine and in which order to play them.)

* With time (credit assignment - there are a number of agents and a number of tasks. Any agent can be assigned to perform any task, incurring some cost that may vary depending on the agent-task assignment. It is required to perform all tasks by assigning exactly one agent to each task and exactly one task to each agent in such a way that the total cost of the assignment is minimized.)

Hybrid
======

There exist hybrid systems, which combine two or more of the three above. An example of a hybrid system would be IBM’s translation system attempts early on. They treated language translation as a machine learning problem, trying to learn the properties of a communication channel and convert it. To translate from French to English and vice versa they used French parliamentary proceedings as their data set, as they are made available in both languages. 

Assignment:
===========

* Soon we'll get an assignment.
* Grade each other’s (double blind) to make a sparse matrix.
* Use this as a dataset for some interesting problems.




#Types of Machine Learning

1. Supervised 
2. Unsupervised
3. Hybrid-RL

Perceptron fits under supervised.

#Perceptron


![alt text](https://dwave.files.wordpress.com/2011/05/qc_ai_diag1b.jpg "Picture of perceptron")

The Perceptron is made up of 3 main components:
Input x<sup>i</sup> (can be many dimensions, 2D - yes/no, higher - what's in a picture? dog, cat? etc.) 

Output: ŷ (what the machine spits out)

Correct: y<sup>(i)</sup> (what the machine should say)

Dials/knobs - weights, w<sup>(i)</sup> is a vector of numbers usually but could be a data structure, or many separate vectors but we'll treat it as one 

You can turn and fiddle with the dials to make things nice, usually by some objective function.
Try hard to get the correct answer - non-objective, this is what it used to be, they used rule of thumb.

Perceptron is an early ML algorithm which fiddles with the dials to get the output correct for a given input.
If you do this repeatedly for a data set you can prove then it will get the correct answer on all of them.

Usually we have some error measure, usually a fraction of what it gets right, we'll call this E, we want this to be low.

E = error over entire training set

E = 1/m(∑(i=0, m-1) E<sup>(i)</sup>) sum over all cases or average, where m is the number of inputs
  = 1/m(∑(i) E<sup>(i)</sup>)
  = <∑<sup>(i)</sup>>    average over i

#Construction of Perceptron

Outputs are binary; 3 options: **True/False**, **1/0**, **+1/-1** 

1/0 useful if you expect lots of zeros e.g. hand-written digit recognition

+1/-1 useful if you're expecting some sort of symmetry and can use this to figure out if it was right or wrong

w<sup>(j)</sup> will be our voltage and θ is our threshold so we end up with equation:
ŷ = ∑(j) x<sup>(j)</sup> * w<sup>(j)</sup> > θ

Basically the inputs x<sup>(j)</sup> are sent in and run through wires going through resistor which are controlled by the dials, these are then added up and the machine then checks if this is greater than the threshold.

Not quite what we want, we want them all to be a whole function

so ŷ = sign(∑(j=1, n) w<sup>(j)</sup>*x<sup>(j)</sup> - θ) this is the Transfer function of a Perceptron. This equation will return -1 if ŷ is less than the threshold and +1 if it's greater than it. 

So what was the motivation behind the Perceptron?
# Neurons

![alt text](http://www.explorecuriocity.org/Portals/2/article%20images/3756/1280px-Neuron.svg.png "Neuron diagram")

In the late 1940’s, it was figured out how neurons work. Alan Hodgkin and Andrew Huxley performed experiments on the giant squid axon, recording ionic currents (for which they got the 1963 Nobel Prize in Physiology and Medicine)


The potential difference between the inside of a neuron and the outside is -80mV. A pulse is transmitted through the axon terminal, causing a chemical reaction and the “gate opens up”. This allows some particles through - ions. Sodium rushes into the neuron, and the voltage increases past the threshold - this is a spike. Now other neurons can figure out what the neuron is doing.

Neurons have "amplifiers" on them called axons, this is because if the signal was too small it would die out and if it was too big it would keep amplifying. 

In a neuron, the maximum spiking rate is 1kHz. This rate is only seen in a dying neuron though. The average spiking rate is ~0.1Hz.

Why don’t we use chemical reactions like this in computers? Cause its slow!


## How to adjust the weights on the machine?

We can change 2 things, the weight or the threshold. 
For our example we'll set theta = -w<sup>(0)</sup> and x<sup>(0)</sup> = 1

ŷ = sign(∑(j=0,n) w<sup>(i)</sup>*x<sup>(i)</sup>)  (we'll call the bit in brackets z i.e. z = ∑(j=0,n) w<sup>(j)</sup>*x<sup>(j)</sup>)

Imagine we have a LEARN button on the machine, if this is pressed it can do 2 things:
If ŷ and y are the same, do nothing. Otherwise adjust the weights accordingly.

| ŷ      | y          | do |
| ------------- |:-------------:| -----:|
| +1     | +1 | nothing |
| -1   | -1   |   nothing |
| -1 | +1      | change w to make z > 0 |
| +1 | -1      | change w to make z < 0 |

In modern techniques you make very small changes to the weights over and over, until you get the desired output. With the Perceptron it would make a big adjustment, so that it got the right answer, but the smallest change possible to get this result.

But how can we do this? Use Linear Algebra. Think of it as symmetrically pushing vectors around.

![alt text](http://www.willamette.edu/~gorr/classes/cs449/Classification/perceptPict-2.gif)

Input space - x ϵ R<sup>(n)</sup> (this is the space the inputs of the machine can "live" in)

In our case it will be a 2D input but it could be more (video, text, audio etc.)

Aside - ours is actually 3 as we set xo = 1, but we can ignore this for maths purposes

Weight space - w ϵ R<sup>(n)</sup>

Output space - binary (yes/no or in our case +1 or -1)

How are we going to change it so that we get the correct response? We've 2 options of what to look at:

For a particular w we look at the input (x), break it up to which we give our to plus or minus 1.

For a particular x we look at weights (w) and see which give plus or minus 1.

In this case w is a separating surface, and we need to turn w. In our graph we'd want to turn it clockwise, but what about in general? It's not always 2D.

We want to move w towards x. 

| ŷ      | y          | do |
| ------------- |:-------------:| -----:|
| +1     | +1 | nothing |
| -1   | -1   |   nothing |
| -1 | +1  | change w to make z > 0   w(t+1) = w(t) + δx |
| +1 | -1      | change w to make z < 0 w(t+1) = w(t) - δx |

w̃(t+1) = w(t) - δ*x

w(t+1) = w̃(t+1)/w̃(t+1)

We want to solve: (w + δx) . x = 0 (i.e. solve for 0)  
(the dots on this and the next few lines represent dot products)

w.x + δ*x.x = 0

w.x + δ||x||^2 = 0

δ = w.x/||x||^2 (this would be just on the separation plane)

δ = - w.x/||x||^2 + d (the plus d ensures it’s just over the separation plane)


To make it work for both cases:

δ = (-w.x/||x||^2 + d)*y

δ = (-w.x/||x||^2 + d)*(y-ŷ/2) (this accounts for the do nothing case

What δ? Should we normalise w? Turns out there’s not much difference, convenient not too big or small.

If the output is wrong, press learn and it'll be right next time.
If it sees all inputs it'll get all of the outputs right.

Aiming for the smallest tilt to get it right.
If there’s no setting to get it right then it won't work.
Assignment
==========

Machine Learning in the News
----------------------------
 
Find a story about machine learning in the popular press (broadly
construed, to include blogs or whatever) and write a summary of the
story, from a technical perspective, about 1/2 page in
length. Information to include would be things like what the problem
they're trying to solve is, how they addressed it, where they got
their data, how much data they had, and what it looked like, why they
considered the problem interesting, etc. Turn it in as a plain text
(or simple markdown format) file. No figures or diagrams or
equations---we'll be doing machine learning with these as a dataset at
that would complicate our lives. Also, be sure to do a spelling
correction, because misspellings will make our lives more difficult
later. Do not include your name in the file; I can get that from
Moodle, and it would be a hassle to strip names off. Remember that
other students will be marking these (in addition to myself) so please
try to give them something interesting to read; something you'd enjoy
reading yourself.

Due: Mon 28-Sep-2015.

(I will process them before class Tuesday, so please have them all in
by 2pm Monday because it is a hassle to slip late ones into the
processing pipeline.)

#Notes from previous years

NOTE : Pictures wouldn't cross over so only text

Perceptron Learning Rule

Input comes in
Checks if correctly classified
	if it is → do nothing
if it’s not → change weights by the minimum amount (rotate w a little bit) so that it would be correctly classified

 

Convergence rule: if w exists, then by cycling through the inputs you’ll eventually find w, where all inputs are classified correctly.

w doesn’t exist if there is even a single outlier. No admissible solution.



 Perceptron:
•         Invented in 1957 at the Cornell Aeronautical Laboratory by Frank Rosenblatt.
•         Funded by the United States Office of Naval Research. Used to distinguish tanks from their surrounding environment.
•         This machine was designed for image recognition: it had an array of 400 photocells, randomly connected to the "neurons". Weights were encoded in potentiometers, and weight updates during learning were performed by electric motors.
•         It was later realized that the perceptron was influenced not only by the shapes of images given for its interpretation but it also was effected by the brightness and was unable to clarify the presence of tanks when there was a different brightness to the time the tank present data was taken.
How does the perceptron work?
 
Figure 1. :  is a graphical illustration of a perceptron with inputs, ...,   and output   (sourced from http://reference.wolfram.com/applications/neuralnetworks/NeuralNetworkTheory/2.4.0.html)
As seen in figure 1 the weighted sum of the inputs and the unity bias are first summed and followed by being processed by a step function yielding the output
 	(x, w, b)= UnitStep (w1 x1 + w¬2 x¬2 +  . . . + wn xn + b)
Where {w1. . . wn} are the weights applied to the input vector and b is the bias weight. Each of the weights are represented by the arrows in figure 1. The UnitStep function is 0 for arguments less than 0 and 1 elsewhere. So   can take values of 0 or 1 depending on the value of the weighted sum. The perceptron can indicate 2 classes corresponding to these 2 input values. While in the training process, the weights (inputs and bias) are adjusted so the input data is mapped correctly to one of the two classes.
Off sample performance more important!!!




Cross validation
Cross-validation, sometimes called rotation estimation, is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. Generally better for a scenario where the goals are predicted, and the test is to see how accurate the prediction is by having a training set and a validation set. The goal of cross validation is to define a dataset to "test" the model in the training phase (i.e., the validation dataset), in order to limit problems like overfitting, give an insight on how the model will generalize to an independent data set (i.e., an unknown dataset, for instance from a real problem). 
A bigger data set presumably gives a better generalization!
figure 1: the line between +’s and –‘s is constantly moving until it fits to a particular position where the margin between the –‘s and +’s are equal.
 A problem with this will be if they aren’t separated appropriately. 3 common problems causing this may be outliers, noise and mislabelled points. It is best to keep the margin between +’s and –‘s as large as possible. 
Margin= distance from separating surface to nearest point, assuming points are correct. 
The closest points to the separating surface are known as support vectors and they are used to help calculate the maximum margin.







Rosenblatt's perceptron

Perceptron Research from the 50s and 60s

https://www.youtube.com/watch?v=cNxadbrN_aI

https://www.youtube.com/watch?v=evVSV43CRk0

training vs testing

regression

stochastic gradient

https://www.youtube.com/user/stanfordhelicopter

Transcription from Lecture
----------------------------------------------
Machine Learning Tree
=====================
<Include machine learning tree>

###Perceptron Equations

> < x <sup>T</sup> x > w = < xy<sup>T</sup> >

> (ɛI + < x <sup>T</sup>x> w) = < xy<sup>t</sup>>

The above is a more robust error measure (Used mainly by statisticians)

Instead of E = <sup>1</sup>/<sub>2</sub> < ||ŷ - y||<sup>2</sup> > we use a lower power
    E = <sup>1</sup>/<sub>2</sub> < ||ŷ - y||<sup>1</sup> >

<Insert demonstration graph>

Another factor that would help in making the system robust would be the removal of outliers from the dataset.

Non-Linear Functions
====================
We'll be moving on from using linear classifiers (Perceptrons) for machine learning to using streaming, non-linear functions. A use case for using these non-linear functions would be in the processing video clips. One training case for a 5 second clip would be ~20mb! Which would include around a million different vectors for machine learning.

The < x <sup>T</sup> x > matrix space is in the order of O(n<sup>2</sup>)

Stochastic Gradient Descent
===========================

> ∇<sub>w</sub>E = <sup>1<sub>0</sub></sup> 0

As is, it's great for finding the minimum for flat curves in the weight-space. However it's more difficult to solve for more complicated shapes.

[Topographic_1]
[Topographic_2]

##Finding the optimum for a more complicated shape.

We undergo a process of iteration in order to find the local minima.

>w(t+1) = w(t) - η∇<sub>w</sub>E

>where η > 0

The function above is known as a naïve gradient descent as we only select one value for η. A better implementations would have more values for η as we descend.
> E = <sup>1</sup>/<sub>2</sub> < ||ŷ - y||<sup>2</sup> >

> ∇<sub>w</sub>E = < <sup>1</sup>/<sub>2</sub> ∇<sub>w</sub> ||ŷ - y||<sup>2</sup> > <- ŷ = f(x;w)

> <text>=</text> < <sup>1</sup>/<sub>2</sub> ∇<sub>w</sub> (ŷ - y)<sup>T</sup>(ŷ - y) >

> <text>=</text> < <sup>1</sup>/<sub>2</sub> ∇<sub>w</sub> (ŷ <sup>T</sup> ŷ) -2y<sup>T</sup> ŷ + y<sup>T</sup>y >

> <sup>dE</sup>/<sub>dw<sub>k</sub></sub> = < <sup>1</sup>/<sub>2</sub> ( <sup>d</sup>/<sub>dw<sub>k</sub></sub>(ŷ <sup>T</sup> ŷ) -2y<sup>T</sup> (<sup>d</sup>/<sub>dw<sub>k</sub></sub> ŷ) ) >

Dimensionality of the ŷ matrix is respective to w.

Average error of whole dataset:

> E = <Ê<sup>(i)</sup>><sub>i</sub> <- Also known as the per sample error average

Gradient descent using this sample error average:

∇<sub>w</sub>E = <∇<sub>w</sub>Ê>
> 
Stochastic gradient descent
***

> w(t+1) = w(t) - η<∇<sub>w</sub>Ê> 

If η(t) ~= <sup>1</sup>/<sub>t</sub>, the algorithm is guaranteed to converge to a local minimum. However, we must constrain η(t) like so:

>O(<sup>1</sup>/<sub>t<sup>2</sup></sub>) < η(t) ≤ O(<sup>1</sup>/<sub>t</sub>)

The SGD algorithm will yield increasingly better results after a 'warm up' period.

We add what we've gathered to a diagram of a perceptron.

[Diagram here]

>s(φ) = <sup>1</sup>/<sub>1+e<sup>-φ</sup></sub>

1) This equation can be considered as a good approximation for current spiking in a biological neuron

2) The following evaluation deals with the output from the perceptron diagram above:
> Ê = <sup>1</sup>/<sub>2</sub>(ŷ - y)<sup>2</sup>

> <sup>dÊ</sup>/<sub>dw<sub>k</sub></sub> =<strike> <sup>1</sup>/<sub>2</sub> . 2 </strike> (ŷ - y) . s<sup>'</sup>(w.x)x<sub>k</sub>

> w<sub>k</sub>(t+1) = w<sub>k</sub>(t) - η(ŷ - y)s<sup>'</sup>(w . x))x<sub>k</sub>

> ∴

> w<sub>k</sub>(t+1) = w<sub>k</sub>(t) - η(ŷ - y) s<sup>'</sup> (w . x) x<sub>k</sub>

η is considered as our learning rate. The smaller the value for η the more stable the system is.


> (s<sup>'</sup>(φ) = s(φ)(1-s(φ)) <--- Epidemic equation)



Semi linear systems
===================

The error function defined below is called the Cross-Entropy function. This matches well with s(φ).

> E = -(y log ŷ + (1-y)log(1-ŷ))

> Minimum ŷ = y
> If y = 0 as ŷ tends 1 then E tends to infinity

In semi linear systems we can define our output ŷ as being a probability of truth

##Logistic Regression
Logistic regression can be defined as the combination of the Cross-Entropy error + Regression






# Calculating the gradient 

Numerical differentiation
dE/dw = (E(w + hei) - E(w)) / h

f'(x) = lim{h->0} ( f(x+h) - f(x) ) / h

programming languages don't usually have a lim construct so let h be a small number

f'(x) = ( f(x+h) - f(x) ) / h

where h = 1/10^9

Efficiency - gradient has m components, so you would have to calculate the above m times. 

Time complexity - O(cal. E).O(cal. m) - It is not efficient

Working with floating point numbers:

1. do not add very small and large numbers.
2. do not subtract numbers of similar sizes.

The above method violates both these rules.

## Back Propagation
Space: Takes up a lot of space.

Operation Count: For each primitive function the derivative must be calculated. At worst this will be a small constant vector more work.

Exact: It is exact up to floating point issues.

See Images/lecture-04 for descriptions of back propagation with unary and binary functions and functions which have one input and two outputs.

## Disjoint Networks
Have a series of networks computing the same problem. If they are all making the same predictions, the probability they are getting the correct result is higher than if they are making difference predictions.






6/10/2015
Backpropagation Networks in the Wild
====================================

Backpropagation Success Stories
-------------------------------

* PapNet http://www.medscape.com/viewarticle/718182_5 http://www.lightparty.com/Health/Papnet.html
* ALVINN http://papers.nips.cc/paper/95-alvinn-an-autonomous-land-vehicle-in-a-neural-network.pdf http://virtuallab.kar.elf.stuba.sk/robowiki/images/e/e8/Lecture_ALVINN.pdf
* RNN Handwriting Generator https://www.cs.toronto.edu/~graves/handwriting.cgi


#PapNet
* Developed in 1980s
* Computer Vision techniques used to centre cells

* Features were extrapolated manually (colour etc.)  
  A level of suspicion score is returned from the system

* Each returned slide would be separated with score attached

**Last trick:** 120 most significant cellular images (Sorted by level of suspicion) presented to human operator
			
**Usual flaws**:  
  * false positives
  * cell not fully stained
  * antibodies attached to it etc.
  
Humans could decipher these flaws easily
		
##PapNet Success
* Much cheaper and accurate than traditional methods
* First billion dollar company using Machine Learning
 
##Aside
* Congress created a law to decrease false-negative Pap smear rates
 
#ALVINN (Autonomous Land Vehicle In a Neural Network)
 Push by US military to create autonomous vehicles  
 **Biggest concern:** convoys being attacked
 
**Idea:** First truck driven by person then followed by bots
 
 **First Trials:** take images and try to describe attributes of the image such as _roads_ etc.
 
Big courses were created out in desert for those autonomous vehicles

##Back-Propagation Method
* Dean Pomerleau (while PhD student at CMU)
  Slapped a camera and laser range on the vehicles
* Early days of back-propagation
  
Pomerleau created a network:  
![ALVINN Architecture](https://raw.githubusercontent.com/GooseyGooLay/Machine-Learing-Images/master/ALVINN%20Architecture.PNG)  
*	8 horizontal scans of 30*32 images
	The Range Finder returns distance of vehicle to the objects
	
* System returns what the steering direction should be  
	Trained using Gaussian bump

ALVINN: Shallow architecture (only one layer of hidden units)

**Problem:** Road intensity feedback (Lightness of road compared to background)

###Aside
Similar system used in Mars Rover project?  
Tom Mathis worked on Mars Rover project also
##ALVINN
* **One trick:** Get training data right  
            	  -Drive vehicle down road correctly once
	
**Problems**  
* Unusual configuration: getting out of parking spot, parallel parking etc.   
                        (Noise on range finder)  
                        (Off angle - not in training set)

##Interpolation vs. Extrapolation
 <pre>"Interpolation - easy, extrapolation - hard"</pre>

![Interpolation/Extrapolation](http://pillars.che.pitt.edu/files/course_12/figures/curve.gif)

* Reason why time-series is hard to create

Pomerleau: made **fake data** - moved road slightly to create new scenarios and train it on those

* Weights can correlate with the colour display from the images

* Inhibited by bright parts on some of the images  
  Excited by others in different areas

* Looks for edges based on contrast of colour in the image  
  Each output is weak evidence (suggestion)  
  "probably a road around this way - evidence for sharp right - evidence against soft left etc."

* Tricky job for person: take all this weighted evidence and making sense of it  

* Vehicle trained to drive on single pathed lane initially  
-Trip from San Fran to Seattle made by the autonomous vehicle  
	(Human exerted control over steering at times needed)

##ALVINN Influence
* **Google cars:** based on this model  
                -new data available like GPS

* **High-end cars:** tiny camera  
                    System: uses above technology to notice if there is imminent crash and then takes control of engine
		                (sees pedestrian etc.)
		                -makes a noise to alert driver

* Single Vehicle Roadway Departure Accidents
  -methodology to decrease these accidents

##Conclusion
* Network Weights Evolve improves after training

One criticism of this style of learning:
* Similar to Black box: hard to figure out what's going on internally  
                        (Will this system drive appropriately during an eclipse?)

#RNN Handwriting Generator
![RNN Handwriting Generator Architecture] (https://raw.githubusercontent.com/GooseyGooLay/Machine-Learing-Images/master/RNN%20Handwriting%20Generator%20Architecture.PNG)

**Recurrent Neural Network** (useful for time-series)

* Output writes with particular style which had been fed in as input  
  -elements of randomness involved

* Network trained on many styles of handwriting
* Style parameter separated at beginning

* Doesn't handle elements not in data set very well


#Training vs. testing (sets)

![Training vs. Testing] (http://i.stack.imgur.com/I7LiT.png)

* Error should go monotonically downwards  
  Only Care about cases which are explicit (minimise one weight's significance and put emphasis on another)

* Validation set should start out with standard error

#Neural Network Halting Problem
* Minimise validation on your training sets

* Optimising rules for small sample set but over time other weights will lose significance when they should
* (**anecdote:** shared interests of people who you are to make happy changes as you suit the needs of a particular sample)

* Particular variables in training set which don't exist overall  
  -leaves us with error

  
  
  

# Lecture 7 Dynamics of Gradient Descent
An important part of Machine Learning is how much time it takes

## Gradient Descent's connection to Physics
The Error function in machine learning is the analogue to Energy in Physics.

Gradient descent in physical terms is like a table wobbling or a pendulum swinging. When the table is deformed there is a restoring force pushing back. Or it is like a pendulum when it is pushed there is a restoring force proportional to the distance it was moved from the point of minimum energy. 
It is not an abrupt transition like letting a table falling. 


## 1 Dimensional situation
We have for a pendulum:
F(x) = - c*x, where F is the force on the pendulum, x is the distance for minimum energy position. 

E(x) = ∫<sup>x</sup><sub>0</sub> F(s) ds
= -c∫<sup>x</sup><sub>0</sub> s ds
= -c s<sup>2</sup> / 2 |<sup>x</sup><sub>0</sub>
=1/2 c x<sup>2</sup>

In a discrete system we have the following formula for gradient descent:
w(t+1) = w(t) - η ∇<sub>w</sub>E

In a continuous physical system we have:
w(t+h) = w(t) - η h ∇<sub>w</sub>E

Rearranging we have:

(w(t+h) - w(t))/h = - η ∇<sub>w</sub>E

Taking the limit as h → 0 we have that:
dw/dt = - η ∇<sub>w</sub>E.

This formula corresponds to a particle going downhill with no momentum (the particle would be said to be in a highly viscous medium)

The steps taken in finding the gradient above are very small, which from a computational point of view is very bad. What we want to do is to take as large of steps as possible.


### How big of a step can we take?
For a 1 Dimension system:
w(t+1) = w(t) - η ∇<sub>w</sub>E
= w(t) - η d/dw(E)
= w(t) - η d/dw(1/2 c w(t)<sup>2</sup>), using formula for E above
= w(t) - η c w(t)
= w(t)(1 - η c)

c and η must both be greater than zero. You can't choose c but can choose η (the learning rate)

We can write the above another way:

w(t) = (1 - c η)<sup>t</sup> w(0)

The limitation on η:

we must have |1- cη|<1 otherwise it would grow and not would not have convergence of the gradient.  From |1- cη|<1 we get that
-1< 1 - cη
⇒ -2 < cη
⇒ η < 2/c, which is called the limit of conversion.

letting η = 1/c will give us the fastest possible convergence of the gradient.

## 2 Dimensional DeCoupled situation
For this situation we have the following Energy formula:

E= 1/2 c<sub>1</sub> w<sub>1</sub><sup>2</sup>+1/2 c<sub>2</sub> w<sub>2</sub><sup>2</sup>.

Since 1/2 c<sub>1</sub> w<sub>1</sub><sup>2</sup> and 1/2 c<sub>2</sub> w<sub>2</sub><sup>2</sup> are independent of one another, the system is said to be decoupled.

We will assume only 1 value of η can be picked.


w= (w1, w2)

∇<sub>w</sub>E= (d/dw<sub>1</sub> (E), d/dw<sub>2</sub>(E))
= (c<sub>1</sub> w<sub>1</sub>,c<sub>2</sub> w<sub>2</sub>)

Looking at the formula w(t+1) = w(t) - η∇<sub>w</sub>E, we get the independent equations:

w<sub>1</sub> (t+1) = w<sub>1</sub>(t) - ηc<sub>1</sub> w<sub>1</sub>(t)


w<sub>2</sub> (t+1) = w<sub>2</sub>(t) - ηc<sub>2</sub> w<sub>2</sub>(t)

and the constraints that η < 2/c<sub>1</sub> and η < 2/c<sub>2</sub>
⇒ η < 2/max(c<sub>1</sub>,c<sub>2</sub>).

We can rewrite the formulas above as:
w<sub>1</sub> (t) = (1- ηc<sub>1</sub>)<sup>t</sup> w<sub>1</sub>(0) and

w<sub>2</sub> (t) = (1- ηc<sub>1</sub>)<sup>t</sup> w<sub>2</sub>(0)


Let c<sub>1</sub> < c<sub>2</sub> then η < 2/c<sub>2</sub> 

Set η= 1/c<sub>2</sub>

w<sub>2</sub> (t) = 0 and

w<sub>1</sub> (t) = (1- c<sub>1</sub>/c<sub>2</sub>)<sup>t</sup> w<sub>1</sub>(0)

Since 1-c<sub>1</sub>/c<sub>2</sub> will be close to 1, then you have to raise t to a very high power to find the optimum gradient. That ratio max(c<sub>1</sub>,c<sub>2</sub>)/max(c<sub>1</sub>,c<sub>2</sub>) is called the convergence limit and it determines the speed of learning. 

However a decoupled situation is not very realistic and most situations are non-quadratic. However near the optimum it looks quadratic.

## 2D Coupled situation
In this scenario we will use Taylor series around the point w*, the point at which the maximum gradient is.

E(w\*+ Δw) = E(w *) + 0 + 1/2 Δw<sup>T</sup> ∇<sup>2</sup><sub>w</sub>E(Δw) + O(⏸Δw⏸<sup>3</sup>).

we get 0 as the second term since ∇<sub>w</sub>E = 0 at w\*

Set H = ∇<sup>2</sup><sub>w</sub>E

For the previous decoupled situation we have:

H = [[∂<sup>2</sup>E/∂w<sub>1</sub><sup>2</sup>, ∂<sup>2</sup>E/∂w<sub>1</sub>∂w<sub>2</sub>], [∂<sup>2</sup>E/∂w<sub>2</sub><sup>2</sup>, ∂<sup>2</sup>E/∂w<sub>2</sub>∂w<sub>2</sub>]] = [[c<sub>1</sub>,0],[0,c<sub>2</sub>]], that is, the 2×2 diagonal matrix with c<sub>1</sub> and c<sub>2</sub> along the main diagonal.

Hence 1/2 Δw<sup>T</sup> H Δw = 1/2 Σ<sub>i</sub> Σ<sub>j</sub> ∂<sup>2</sup>E/∂w<sub>i</sub>∂w<sub>j</sub> Δw<sub>i</sub> Δw<sub>j</sub>.

Diagonal Matrices are easy to work with. We can choose a new coordinate system (a new basis) such that we always get a diagonal matrix through the use of eigenvectors.

∇E(w\* + Δw) = H Δw + O(⏸Δw⏸<sup>2</sup>)
⇒ ∇E(w\* + Δw) ≅ H Δw

Let v<sub>i</sub> be an eigenvector of H then Hv<sub>i</sub> = λ<sub>i</sub> v<sub>i</sub> and we can express Δw in the eigenbasis:

Δw = Σ<sub>i</sub> b<sub>i</sub> v<sub>i</sub>

Then H Δw = H Σ<sub>i</sub> b<sub>i</sub> v<sub>i</sub> = Σ<sub>i</sub> b<sub>i</sub> H v<sub>i</sub> = Σ<sub>i</sub> b<sub>i</sub> λ<sub>i</sub> v<sub>i</sub>.

Hence the formula for b<sub>i</sub>(t+1) = λ<sub>i</sub> - η λ<sub>i</sub> b<sub>i</sub>(t) with η < 2/(max<sub>i</sub> λ<sub>i</sub>)

The "condition number" of H is λ<sub>max</sub> / λ<sub>min</sub>.







Lecture 8
Notes

When we are close enough to the optimum the formula becomes quadratic and convergence rate depends on the ratio λ<sub>max</sub> / λ<sub>min</sub>


Grading descent equation from previous lectures:

> w(t+1) = w(t) - η ∇E (where η is learning rate)

Depending on our chosen learning rate we can have different outcomes:

a) learning rate is too big and convergence will never happen - with every step of descent we will move further and further from the optimum:

b) learning rate is too small, convergence will happen but this will be a slow progress and will take a lot of iterations of the algorithm

![Choice of learning rate](http://sebastianraschka.com/Images_old/2015_singlelayer_neurons/perceptron_learning_rate.png)

c) well-chosen convergence rate will converge and do so reasonably quickly

To obtain such optimal learning rate we will introduce new variable - α, for momentum, such that


> 0 ≤ α < 1

(at zero we have no momentum at all, and at 1 momentum will not stop at the minimum, we need some friction to slow it down once we've reached our goal)

If we add this new factor our formula becomes:

> w(t+1) = w(t) - η ∇E + α (w(t) - w(t-1))

Simplify:

> w(t+1) - w(t) = - η∇E + α(w(t) - w(t-1))

Rewrite w(t+1) - w(t) as Δw(t) and substitute it into the formula above:

> Δw(t) = - η ∇E + α Δw(t-1)

where for stability η < 2 / λ<sub>max</sub>

There has to be a balance in setting the momentum to apply optimally to both λ<sub>max</sub> and λ<sub>min</sub>. If λ<sub>max</sub> is a lot bigger than λ<sub>min</sub> we calculate momentum using the following formula:

> Δb<sub>min</sub> = - η∂E/∂b<sub>min</sub> + αΔb<sub>min</sub>

> Δb<sub>min</sub> - αΔb<sub>min</sub> = - η∂E/∂b<sub>min</sub>

> (1 - α)Δb<sub>min</sub> = - η∂E/∂b<sub>min</sub>

> Δb<sub>min</sub> = - η/(1 - α)∂E/∂b<sub>min</sub>

if α is too high it would affect our λ<sub>max</sub> which will have it overshoot the minimum.

However, in practice, for big data sets there are too many calculations to use batch grading descent. 
Instead, another option is to use stochastic grading descent:

> Δw(t) = - η ∇Ê 

(where Ê is ∇E plus some bounded amount of zero-mean noise)

- η has to go to zero, slowly enough that we can get rid of the noise.

> Σ<sub>t=1...∞</sub> η(t) = ∞ 

(to have enough momentum), but also, for descent to be fast enough

> Σ<sub>t=1...∞</sub> η(t)<sup>2</sup> < ∞:

so

> O(1/√t) < η(t) ≤ O(1/t)

In practice stochastic descent is not used either. We can't always get an optimum, for an algorithm that uses real life changing data a good approximation is what we aim for.

All gradient descent methods are weak, it is much better to analytically find the optimum and just "jump" there.

## Support Vector Machines (SVMs):

MLP require a lot of tuning. They are hard to implement and there's a lot of decisions that we have to make.
New approach - SVM (developed by [Vladimir Vapnik]( https://en.wikipedia.org/wiki/Vladimir_Vapnik) and his colleagues)
Some of the SVM logic can be added to MLPs to enhance their performance.
Data example:

![data example for fitting linear classifier](images/lecture-08/separating-lines.png)

For data like this we can fit many possible linear thresholds that will predict different values. If we had to choose one, we could find a place, such that:
1) classifies data correctly
2) positioned in such a way that the nearest square and the nearest circle are as far away from it as possible.

![data example for fitting SV](images/lecture-08/optimal-hyperplane.png)

Its position depends solely on data points touching the margin (the support vectors).


SVM - linear classifier with maximum margin. (The bigger the margin, the better)







20/10/2015

#Support Vector Machine (SVM)
<b>Last Lecture</b>: Maximum margin problems
* Introduction of soft margin  
When there is mislabelled data; a hyperplane is introduced to cleanly split data and maximise margin distance

* Problem based on linear classification  
Margin is straightforward to calculate in linear case but not when the problem's non-linear:

![Non-linear margin] (http://www.blaenkdenum.com/images/notes/machine-learning/support-vector-machines/x-space-non-linear-svm.png)

##Kernel Trick
* Map observations to a higher dimensional space using a <b>Kernel Function</b>:  
![Kernel Function](https://upload.wikimedia.org/math/9/c/b/9cbd072b356b4cb62afceef088c751dd.png)  
e.g.  
<pre>φ: ℝ<sup>10<sup>6</sup></sup>⟼ ℝ<sup>10<sup>100</sup></sup></pre>  

* φ(x) - Intractable (heard to calculate by itself)  
  Analogy - GPU input vector which you cannot alter once it is being processed

##Kernelize the Algorithm
* Instead of operating in input space - change x's to φ(x<sup>i</sup>)'s to move to feature space

<b>ω</b>: primal/single representation of the vector  
<b>α</b>: dual representation of the vector  
<b>ξ<sub>i</sub></b>: slack parameter  
ω,θ,ξ - penetration variables (penetrate margins)  
y<sup>(i)</sup> = ±1 for either "yes"/"no" class of data points
<pre>
Minimise<sub>(ω,θ,ξ)</sub>  
              <sup>1</sup>/<sub>2</sub>‖ω‖<sup>2</sup> + cΣ<sub>i</sub>(ξ<sub>i</sub>)
Subject to:  
              y<sup>(i)</sup>(ω·x<sup>(i)</sup> - θ) ≥ 1 - ξ<sub>i</sub>
And:  
              ξ<sub>i</sub> ≥ 0
</pre>
Math Breakdown:
<pre> ω= Σ<sub>i</sub>(α<sub>i</sub>y<sup>(i)</sup>φ(x<sup>(i)</sup>)
  
‖ω‖<sup>2</sup> = ω·ω = (Σ<sub>i</sub>(α<sub>i</sub>y<sup>(i)</sup>φ(x<sup>(i)</sup>))·(Σ<sub>j</sub>(α<sub>j</sub>y<sup>(j)</sup>φ(x<sup>(j)</sup>))
    
          = Σ<sub>i,j</sub>(α<sub>i</sub>α<sub>j</sub>y<sup>(i)</sup>y<sup>(j)</sup>(φ(x<sup>(i)</sup>)φ(x<sup>(j)</sup>)))
                
          = Σ<sub>i,j</sub>(α<sub>i</sub>α<sub>j</sub>y<sup>(i)</sup>y<sup>(j)</sup>k(x<sup>(i)</sup>,x<sup>(j)</sup>))
                
          = α<sup>T</sup>(y<sup>(i)</sup>y<sup>(j)</sup>k<sub>i,j</sub>)<sub>i,j</sub>
</pre>

<pre>Substitute ω·φ(x<sup>(i)</sup>)

           = Σ<sub>j</sub>(α<sub>j</sub>y<sup>(j)</sup>φ(x<sup>(j)</sup>)φ(x<sup>(i)</sup>))
          
           = Σ<sub>j</sub>(α<sub>j</sub>y<sup>(j)</sup>k(x<sup>(i)</sup>,x<sup>(j)</sup>))
</pre>
###Kernelized Algorithm
<pre>
Minimise<sub>(α,θ,ξ)</sub>  
              <sup>1</sup>/<sub>2</sub>α<sup>T</sup>(y<sup>(i)</sup>y<sup>(j)</sup>k<sub>i,j</sub>)<sub>i,j</sub> + cΣ<sub>i</sub>(ξ<sub>i</sub>)
Subject to:  
              y<sup>(i)</sup>(Σ<sub>j</sub>(α<sub>j</sub>y<sup>(j)</sup>k(x<sup>(i)</sup>,x<sup>(j)</sup>)) - θ) ≥ 1 - ξ<sub>i</sub>
And:  
              ξ<sub>i</sub> ≥ 0
</pre>

* Dual representation (α) used for quadratic programming problems as opposed to primal representation (ω)

##Kernel Function
####Mercer's Theorem
<b>Pre-condition:</b>  
If k is <i>symmetric:</i>   
<pre>k(u, v) = k(v, u)</pre>
,<i>non-negative definite</i>:  
![non-negative definite kernel] (https://upload.wikimedia.org/math/7/9/e/79e0f0a14643312d46347a004e688ef7.png)  
for all finite sequences of points x<sub>1,...,</sub> x<sub>n</sub> of [a, b] and all choices of real numbers c<sub>1,...,</sub> c<sub>n</sub>  
<b>Post-condition:</b>  
<pre>⇒ ∃ φ s.t. k(u,v)=φ(u)·(v)</pre>

<b>Examples:</b>  
Identity Kernel:
<pre> k(u,v)=u·v</pre>  
* takes O(n) work in n-space  

<pre> k(u,v) = (u·v)<sup>2</sup>

          = (Σ<sub>k</sub>(u<sub>k</sub>v<sub>k</sub>))<sup>2</sup>
          
          = (Σ<sub>k</sub>(u<sub>k</sub>v<sub>k</sub>))(Σ<sub>k<sup>'</sup></sub>(u<sub>k<sup>'</sup></sub>v<sub>k<sup>'</sup></sub>))
          
          = Σ<sub>k,k<sup>'</sup></sub>(u<sub>k</sub>u<sub>k<sup>'</sup></sub>)(v<sub>k</sub>v<sub>k<sup>'</sup></sub>)
          
          = Σ<sub>k,k<sup>'</sup></sub>φ(u)<sub>k,k<sup>'</sup></sub>φ(v)<sub>k,k<sup>'</sup></sub>

</pre>

###Polynomial Kernel
For degree-d polynomials, the polynomial kernel is defined as:  
![polynomial kernel] (https://upload.wikimedia.org/math/e/0/e/e0e6e2ac260502f8818fb8c55cec2227.png)  
where x and y are vectors in the input space and c ≥ 0 is a free parameter trading off the influence of higher-order versus lower-order terms in the polynomial.

<pre>φ: ℝ<sup>n</sup>⟼ ℝ<sup>n<sup>p</sup>/≈p!</sup></pre>

* When we used quadratic kernel we dropped all linear terms

* No arguments about which kernel function to use as you can always use them both (add them up)
Example:  

<pre> k(u,v) = (u·v + 1)<sup>2</sup>

        = (Σ<sub>k</sub>u<sub>k</sub>v<sub>k</sub> + 1)(Σ<sub>k<sup>'</sup></sub>u<sub>k<sup>'</sup></sub>v<sub>k<sup>'</sup></sub> + 1)
        
        = Σ<sub>k,k<sup>'</sup></sub>u<sub>k</sub>u<sub>k<sup>'</sup></sub>v<sub>k</sub>v<sub>k<sup>'</sup></sub> + 2Σ<sub>k</sub>u<sub>k</sub>v<sub>k</sub> + 1
</pre>     

##Gaussian Process
* <b>Gaussian Kernel</b>:  
<pre>k(u,v) = e<sup>-d|u-v|<sup>2</sup></sup></pre>
* Can be used to compute similarities between images
* Fee for using this: maps to infinite dimensions

##Kernel Function applications
* Find similarities between two pieces of text

*When we finished the optimisation above:  
Problem: there were no x's left, just i's and j's

When we want to embed SVM in your system (sneeze function in camera):
<pre>ω·φ(x)⩼ θ

Σ(α<sub>i</sub>y<sup>(i)</sup>φ(x<sup>(i)</sup>)·φ(x))
=Σ<sub>(s.t. α<sub>i</sub>≠0)</sub>(α<sub>i</sub>y<sup>(i)</sup>k(x<sup>(i)</sup>,x))

most α<sub>i</sub> are zero!
</pre>

We only need to store the support vectors of people sneezing from the training set  
Only download these into the camera:  
e.g. <sup>200</sup>/<sub>1000</sub> training cases

##SVM Conclusions
Popular kernels: quite robust  
-Reasons why people like SVM instead

<b>Positives:</b> 
* Beautiful Math (Kercher's...)
* SVM depends on number of support vectors  
  - can work in higher-dimensional space as only looks at subset of vectors
* Turn Key  

<b>Criticism:</b>
* Glorified template matching







2/11/2015

Lecture 11
Notes

Unsupervised Learning
#### Clustering


[Linear vector quantization](https://en.wikipedia.org/wiki/Learning_vector_quantization) (LVQ) aka k-means

Example.
You are given a task of calculating an average weight of a mouse in some mouse colony.
How would you approach it? You could measure the weight of every mouse, sum it up and then divide the total by a number of mice measured.

If there are too many mice to measure, you could, perhaps, measure some representative sample of the mouse population and calculate the average based on the sample.

But what would you do if the average was dynamic, for instance, the mice are a subject of the experiment that affects their weight and we want to track the changes to the average weight? Keep in mind, your solution needs to be realistic in terms of data storage we can designate for our calculations.

To calculate a running average we simply need to keep track of the running total and the current count:

> a<sub>1</sub>, ..., a<sub>t</sub> are weights

> S<sub>t</sub> = Σ<sub>i=1...t</sub> a<sub>i</sub> is a total weight

and then the average weight is calculated as follows:

> A<sub>t</sub> = S<sub>t</sub>/t

if we need to add another mouse's weight the next day we update out total and our count:

> S<sub>t+1</sub> = S<sub>t</sub> + a<sub>t+1</sub> 

> A<sub>t+1</sub> = S<sub>t+1</sub>/t+1

In our case, however, if we want to track the changes in the average weight, we may want to only calculate the average weight over the last month.

We want to disregard old data and give more weight to the new data. To do so we can use decaying sum:

Decaying estimate:

> S<sub>t</sub> = a<sub>t</sub> + 0.98 a<sub>t-1</sub> + (0.98)<sup>2</sup> a<sub>t-2</sub> + ... = Σ<sub>j=0...∞</sub> (0.98)<sup>j</sup> a<sub>t-j</sub> 

We call (0.98) in the following example decaying constant and we can denote it as α, s.t 0 < α < 1

We adjust our formulas accordingly:

> S<sub>t+1</sub> = a<sub>t+1</sub> + 0.98 S<sub>t</sub>

> A<sub>t</sub> = S<sub>t</sub>/Σ<sub>j=0...∞</sub> (0.98)<sup>j</sup>

Now let’s say in our example we have 2 subpopulations of mice and they differ by weight. In this situation we would like 2 averages, each calculated for a separate subpopulation.

How can we begin to sort them?

![bimodal histogram](images/lecture-11/bimodal histogram.png)

Ideally, we would like to be able to classify a mouse as belonging to one or the other subpopulation and only update the relevant subpopulation average with that weight. But we don't know, when we are given a mouse, which class of mice it belongs to. One algorithm gives us a way around it by creating cluster centres and using the cluster centres for classification.

##### [k-means Algorithm](https://en.wikipedia.org/wiki/K-means_clustering)

Using this algorithm we associate the value of every new sample with the value of the cluster centre closest to it.

 loop:
> read in x<sup>t</sup> 

> calculate ĵ = argmin<sub>j</sub> ‖ x<sup>t</sup> - w<sup>j</sup> ‖

> w<sup>ĵ</sup> ← α w<sup>ĵ</sup> + (1 - α) x<sup>t</sup> 

And as we add this new value to some cluster, we update the average value of that cluster and move w<sup>j</sup> closer to the added value.

As the algorithm progresses, some clusters will starve, some will crawl towards each other, they will not be quite as they were during the initialisation, but resulting clusters will be fairly accurate. 

If we initialise many more clusters than we expect we need, and if they are quite spread out, we can arrive to the correct positioning of cluster centres by throwing away starved ones and joining the ones that are close together.

Usages of k-means:

One example of using this algorithm could be recognition of hand written digits.

Another one - speech recognition.

When the system analyses speech for each bin it gets a vector (of ~ 128 dimensions) to represent frequencies. It is hard to compute things using 128 dimensions, so instead of this vector analysis works on cluster indices generated for the vectors. When a new user starts using the system, the clusters get updated to suit.

In clean form, however, clustering is rarely used.

It can be used for cell sorter to find abnormal features in tissue, or in biology, clustering samples into species, or purchase history. In real life clustering is combined with other algorithms to improve its accuracy. 
Factors to consider when using clustering algorithms:

- is clustering going to converge?
- will the pull factor lessen with time and will the centres move less?
- will the resulting clusters be correct?
We can use [Voronoi](https://en.wikipedia.org/wiki/Voronoi_diagram) diagram to split our space into planes where the points lying in the same plane correspond to its cluster centre:

![Voronoi diagram](images/lecture-11/voronoj_diagram_clustering.png)

To find cluster centre w<sup>j</sup> amongst many cluster centres such that x is closest to w<sup>j</sup> we minimise error E using stochastic gradient on clustering function:


> E = Σ<sub>t</sub> ‖ x<sup>t</sup> - w<sup>ĵ</sup> ‖<sup>2</sup>,

where 
> ĵ = argmin<sub>j</sub> ‖ x<sup>t</sup> - w<sup>j</sup> ‖,

hence
> E = Σ<sub>t</sub> minΣ<sub>j</sub> ‖ x<sup>t</sup> - w<sup>j</sup> ‖<sup>2</sup>









3/11/2015

#Bayes' Rule, Coin Toss and K-Means

The proportion of tall people with cancer > Population of people with cancer within entire population

http://www.futilitycloset.com/2015/10/19/more-fun/

Q: If tall people are more likely to get cancer, then are people who get cancer more likely to be tall?

<a href="url"><img src="https://raw.githubusercontent.com/barak/mu-cs401-f2015/0c97806b39f09dd012d666c71538601be86cb2fd/images/lecture-17/Cancer-problem.png" height = "280" width = "598"></a>

So, yes people who get cancer are more likely to be tall.

##Bayes' Rule
<a href="url"><img src="https://raw.githubusercontent.com/barak/mu-cs401-f2015/0c97806b39f09dd012d666c71538601be86cb2fd/images/lecture-17/Bayes-rule.png" height = "227" width = "607"></a>

Imagine a machine, with various parameters, that produces data randomly, and a database of some kind of documents.

When run, it produces a one page document. 

There is a very small chance that it will match a document already in our database.

However, if we set the parameters a certain way, our machine may produce a document similar to docs in the database.

How do we set these parameters properly?

##Setting the Parameters on our documents machine

<a href="url"><img src="https://raw.githubusercontent.com/barak/mu-cs401-f2015/0c97806b39f09dd012d666c71538601be86cb2fd/images/lecture-17/Coins.png" height = "791" width = "1144"></a>

##Discrete Distribution

Consider a coin flipping machine, with a 0-1 probability parameter.
The machine can either produce 0, for heads, or 1, for tails.

It has a binary dataset (y1, ..., yn)

And the machine outputs (ŷ1, ..., ŷn)

In a dataset of ~100, the chances of ŷ = y is vanishing small.

Note that we can store vanishing small numbers as logarithm.

<a href="url"><img src="https://raw.githubusercontent.com/barak/mu-cs401-f2015/0c97806b39f09dd012d666c71538601be86cb2fd/images/lecture-17/bernoulli.png" height = "214" width = "702"></a>

If the machine has memory, then every coin flip will be like taking marbles from a bag, and not replacing them 

-> every previous result will affect the next result.

However, this is not how coin tosses work, every coin toss outcome is independent, so our machine has no memory.

##K-means
K-means is an unsupervised learning algorithm that classifies a given data set into clusters.

We can imagine the dataset is produced by a machine that will populate a graph with data.

The machine has three inner machines, each of which will produce a data point that is around a certain area, but has some degree of randomness or noise.

Using K-means, we want to look at this data set and find out which points were created by which machine.

We will look at K-means further in Lecture 13.







Mixture Model
=============

#Two iterations of:
* (Data, Assignments) ----m-step----> Model
* (Data, Model)       ----e-step----> Assignments

Hidden Markov Model
===================
* Finite state machines
* Baum-Welsh speech recognition for NSA
* Outputs can be associated with states or transitions
* Bernoulli trial and speech recognition examples
* π = probability distribution for the start state

#What makes it "hidden"?
* States are hidden - only have the output
* Goals (given output):
  * Reconstruct sequence of states
  * Determine which model produced it [Bayes' Rule](https://en.wikipedia.org/wiki/Bayes%27_rule)

#Problems:
1. (output seq, HMM)  --infer--> sequence of states
2. (output seq, HMM)  --infer--> P(output seq | HMM)
3. (output seq(s))    --infer--> HMM (figure out the parameters of HMM) (EM steps)

#Markov Property
* Probability of the next state is independent of previous states - only depends on the current state.

##Cheat Sheet
* Tables drawn for:
  * Forward algorithm (alpha)
  * Backward algorithm (beta)

* http://barak.pearlmutter.net/misc/hmm.pdf





Coupled Hidden Markov Model
=============

<HMM image>

HMM can have discrete values

No efficient algorithm is known for the Coupled HMM, 
there is known algorithms for single HMM

Multiscale Quadtree
=============

-Images have multiscale properties, i.e. the statically probability of the distribution remains the same when viewing the image when zoomed in or viewing it full sized.

Stereo fuse (Stereo vision) - Using two camera that are close side-by-side position for depth perception, gives uncertainly which is important, e.g. we know where a ball might land after thrown but can't tell for sure where it'll land but the general area, i.e. a gust of wind could change the trajectory at the last second.

Encoding on noisy channel

m > n
<pre>
   m                   n (message)
 |                  | | 0 |
 |                  | | 1 |
 |                  | | 1 |
 |                  | | 0 |
 |                  | | 1 |
 |                  | | 1 |
 |                  | 
 |                  |

 | Parody (XOR of all bits)|

</pre>

Message matrix < m > | n - message ^ 
Parody (XOR of all bits)

Best matrix for encoding messages on a noisy channel 

<pre>
   m                   n (message)
 |                  | | 0 |
 |   sparse,        | | 1 |
 |   random,        | | 1 |
 |   3 1's per row  | | 0 |
 |   max            | | 1 |
 |                  | | 1 |
 |                  |
 |                  |
</pre>

Above matrix m is the only known one to approach Shannon’s limit of transmission


A Graphical model allows us to use inference to get x

 ( x1 ) → ( x2 ) → ( x3 ) ...... (  ) → (  )
     ↓            ↓             ↓               ↓        ↓
 ( y1 )   ( y2 )    ( y3 )        (  )   (  )


These tables have P(V = V<sub>i</sub> | A<sub>i</sub> = a<sub>i</sub>, A<sub>2</sub> = a<sub>2</sub> .....)

          #v * #a2 * #a3 * ...... * an 

Coin flip table

| V  | a<sub>1</sub>  | a<sub>2</sub>  |
|---|---|---|
|   | 0 | 0 |
|   | 0 | 1 |
|   | 1 | 0 | 
|   | 1 | 1 | 


directed graphical model -> undirected graphical model
Using energy instead of probability 

given ancestors of node, find node

write byte  read byte

  ( x )  ->  ( y )

P (y | x)

| P(y|x)  | y  | x  |
|---|---|---|
| .9  | 0 | 0 |
| .1  | 1 | 0 |
| .2  | 0 | 1 | 
| .8  | 1 | 1 | 

table for directed model

Energy Model
=============

Pxy
P00 = 0.45
P01 = 0.05
P10 = 0.1
P11 = 0.4

P(X, Y) = P(T|X)PX

P(α) = 1 / z * e - (Eα) / x

set T = 1

log Pα = -log(z) - Eα
Eα = -log(z) - log(Pα)
z = Σα^(e-(Eα)/T)

Pα/Pβ 1/z(e) - (Eα)/T    = e^(-Eα+Eβ) = -Eα + Eβ = log (Pα)/Pβ = log(Pα) - log(Pβ)
      ________________
	1/z(e)-(Eβ)/T

log (P00/P01) = -E00+E01

E01 = E00 + log ( P00/P01 )







Intro: Previously on CS401...
======

Machine Learning theorists have a probabilistic perspective. Previously we have looked at probabistic algorithms such as:

* Bayes rule
    * P(A|B) = P(B|A) P(A) / P(B)
    * relates the odds of event A1 to event A2, before and after conditioning to another event B.
* K-means clustering
    * unsupervised learning algorithm that classifies a given data set into clusters.
* Back propagation
    * calculates the gradient of a loss function with respect to all the weights in the network.

Now: Probabalistic Estimation With Confidence Interval
======

## Augmenting the backpropagation network

### Nice to know how certain our output is

Gaussian distribution of output?

[image placeholder]

[derivation placeholder]

## Unsupervised Learning

* Data fit to probability distribution
* Graphical models
    * graph theory
    * inference is hard for graph models

[image placeholder]








How to sample a Graphical Model
==========

The Product-Sum algorithm is an algorithm for computing for sampling a graphical model. It works well if the graph has a tree like structure, but in other situations it isn't. It is sometimes called a message passing algorithm. 

* Add in extra nodes into your graph called "factors" in the middle of each arrow. (The original nodes are called variables). Also you can drop the directness of the graph
* This is now a bipartite graph (i.e. one which can be coloured with only 2 colours and all adjacent nodes have different colours. Associate numbers with all nodes
* Alternate updating these numbers. The Factors get updated by products of terms and the Variables get updated by sums of terms.

*In Practice* We have made some observations (e.g. characters) and we want to figure out the marginal probabilities in the graphical model (e.g. the word someone was trying to type). Finding the full joint distribution is too hard of problem for graphical models 

* We associate a function f(x<sub>i</sub>, x<sub>j</sub>) with the factor between x_<sub>i</sub> and x<sub>j</sub>.
* Hence we can say P(x<sub>i</sub>,...x<sub>n</sub>)= &#928;<sub>j</sub> f  <sub>j</sub>(x<sub>a(j)</sub>,x<sub>b(j)</sub>).
* Update the factors with a message called v. v is a message from a variable x<sub>n</sub> to x<sub>m</sub> and it represents the conditional probability P(x<sub>m</sub>| all the children of n)


Problems
--------------------
With a tree graph, the product-sum algorithm will converge. However with a chain of loop like structure old information will get amplified. If chains are long or there is some sort of attenuation this is not really important, but if the chains are short the probabilities you get will be wrong.  

[Boltzmann Machine](https://en.wikipedia.org/wiki/Boltzmann_machine)
================
*Problem* Given the structure and lots of samples from Graphical models, can we figure out the relationships between them? This is a learning problem. 

From Boltzmann Machine we will deal with binary variables. The graph is separated into 2 parts, the hidden variables and visible variables. The links in the graph encode a complicated probability distribution. The hidden part is supporting structure for the visible.

Let E=Σ<sub>i<j</sub>s<sub>i</sub> s<sub>j</sub> w<sub>ij</sub>.

Let α be some configuration of the system. 
P(α) is proportional to e<sup>-Eα/t</sup>

Pick s<sub>i</sub> and consider it changing its state. 

Calculate E<sub>si=1</sub> and E<sub>si=0</sub>

<!-- Some formulas go here  --> 

<!-- some graphs go here --> 

The parameter T can be viewed as temperature 

A High T gives: 

<!-- graph goes here -->

A Low T gives 

<!-- graphs goes here -->

let α be a configuration of the visible states and β be a confuguration of hidden states 

<!-- Some formulas go here  -->

We want to calculate: 

<!-- Some formulas go here  -->










#Matrix decomposition
___

##Recap
___
+ Graphical models and bayes nets => supposedly the future of ml.
+ However inference from graphs is considered an intractible problem.



##Matrix decomposition 
___

+In general matrix decomposition is the factorization of a matrix into a product of matrices. In particular we talk about non-negative matrix 
factorization(NMF), that is we factorize a matrix into two matrices such that all three matrices are element wise non-negative.

+Many problems can be rephrased as matrix decomposition and thus it is a handy tool to have in the machine learning shed.  
  
+Data sets such as images etc. fall into the line of fire of NMF 

###Example
___
Consider the 2D data set 

![Alt text](images/lecture-18/2DData.jpg)

where y1 and y2 are drawn from gaussian (normal) generators as follows:  

	+y1 ~ g(0;1^2)
	+y2 ~ g(0;2^2)

where a gaussian distribution is parameterised as g(mean; std. dev)

we also have the following energy functions for y1 and y2:  

	+E(y1) = y1^2
	+E(y2) = (y2^2)/4

(for those who are wondering what all this talk of energy functions is about and what they have to do with stats [this](http://www.askamathematician.com/2010/02/q-whats-so-special-about-the-gaussian-distribution-a-k-a-a-normal-distribution-or-bell-curve) might help.)

We can visualise the data using a histogram:  
![Alt text](images/lecture-18/hist.jpg)
  

or with a scatter plot:  
![Alt text](images/lecture-18/scatter.jpg) 


The physics analog here is the distribution of molecules in 2 different rooms. We want to consider them seperately, then consider the joint distribtution of molecules across the two rooms. 
  
Thus the joint energy of the two samples is:  
  
 	E(y1,y2) = (y1^2) +  (y2^2)/4  

Then we can get the probability of some X across the two gaussians:  

	p(x) = 1/Z(e^-(E(y1,y2)))  
	where Z is the partition function(see previous lectures)  

##Non axial parallel example 
___  
In the case where the distributions are non-axial parallel. For example:  

![Alt text](images/lecture-18/nonaxialp.jpg)  
  
There are 2 sources of variation in the data given by ci, where:  
	||ci|| = 1 //ci is a unit vector  
	 λi = amount of std. dev. in direction i

Thus we have another energy function, that describes a 2D gaussian with arbitrary direction:
	![Alt text](images/lecture-18/2dgaussenergy.gif)  

	where:  
![Alt text](images/lecture-18/orthovectconstrain.gif)   // [i=j] knuth notation, returns 0 or 1

##Expanding to n-dimensional gaussians
___  
  
How do we expand this concept to n-space?  

![Alt text](images/lecture-18/ndimengaussianorientation.gif)


Include the lambda's:  
![Alt text](images/lecture-18/ndimenlambdas.gif)  
where  
![Alt text](images/lecture-18/C.gif)  
  

When the distribution(s?) are axial parallel we have: 

![Alt text](images/lecture-18/axialpmatrix.gif)  
  
when the the distribution(s?) are not axial parallel the matrix gives you the orientation of the distribution(s?)  
___  
  
Going back to our 2D data set, we have x expressed as a sum:  
![Alt text](images/lecture-18/2ddataassum.gif)

##Example  
___
say we have an 8 * 8 matrix X, a 64 * 12 matrix A (the column joined ci's) and a 12*M matrix B  
we want to decompose X (approximately)  S.T:  
![Alt text](images/lecture-18/adotb.gif)  

we find:  
![Alt text](images/lecture-18/argmindecomp.gif)  
i.e. the least squares approach  
with the constraint that:  
![Alt text](images/lecture-18/constraintonA.gif)  
  
(aside: all major contributions to the field of statistics have been by pyschologists, not statisticians.)  
  
 What if we relax the above constraint (i.e. that the ci's are pairwise orthogonal)?  
 => No unique solution (to what???)  

 For example consider the distributions:  

![Alt text](images/lecture-18/noorthconstraint.jpg)  
  
  
Usually only one source of variation is non-zero in this case.
This allows for independent component anlaysis.










# More Matrix Decompositions

## PCA and SVD

Note readable intro to PCA:
http://efavdb.com/principal-component-analysis/

## Independent Components Analysis (ICA)

https://en.wikipedia.org/wiki/Independent_component_analysis

http://www.inf.fu-berlin.de/lehre/WS05/Mustererkennung/infomax/infomax.pdf

## Nonnegative Matrix Factorization (NMF)

https://en.wikipedia.org/wiki/Non-negative_matrix_factorization

http://www.nature.com/nature/journal/v401/n6755/full/401788a0.html

http://www.columbia.edu/~jwp2128/Teaching/W4721/papers/nmf_nature.pdf









Independent Component Analysis 
=======================
[Wikipedia Article on ICA](https://en.wikipedia.org/wiki/Independent_component_analysis)

####Example:
We have a group of people betting on a game. Our objective is to minimise the loss we make while betting.

* There are multiple rounds of betting.
* You must bet on every game.
* One person in that group is an expert and is always correct.

**Solution:** Find the expert as quickly as possible.

x<sub>i</sub>(t) = prediction of the ith person in game t
w<sub>i</sub>(t) = (boolean value) whether we listen to the ith person's advice
z(t) = the winner.

Idea: Bet on the side which most people bet on. Only consider the bets of those people who have always been correct up to this point.
 > y(t) = [ Σ<sub>i</sub> (w<sub>i</sub>(t) x<sub>i</sub>(t)) > 1/2 Σ<sub>i</sub> w<sub>i</sub>(t) ]

Update the weight associated with each person (only those who have always been right before this game). If they guessed incorrectly in this game set their weight to 0 else 1. 
 > w<sub>i</sub>(t+1) = w<sub>i</sub>(t) [ x<sub>i</sub>(t) = z(t)]

Regret : How much we will lose before we find the expert. 
Regret <= log<sub>2</sub>n 

**Update Problem:**  No one is always correct but some people are better then others at guessing.

w<sub>i</sub>(t) = weight associated with the ith persons advice.

Idea: Reduce the weight an individual has by 1/2 if they bet incorrectly. Keep their weight the same if they guess correctly.

> w<sub>i</sub>(t+1) = w<sub>i</sub>(t) ( 1/2 [x<sub>i</sub>(t) = z(t)]  + 1/2)












# Sparse Learning and Boosting

## Winnow

https://en.wikipedia.org/wiki/Winnow_(algorithm)

## Strong Learnability

### Probably Approximately Correct (PAC)

https://en.wikipedia.org/wiki/Probably_approximately_correct_learning

Almost certainly generalizes really well

## Weak Learning

Has a slight chance of generalizing slighly better than chance.

## Boosting

https://en.wikipedia.org/wiki/Boosting_(machine_learning)

### ADAboost

https://en.wikipedia.org/wiki/AdaBoost









# Stepping on the Curve

## ROC Curves

https://en.wikipedia.org/wiki/Receiver_operating_characteristic

https://www.google.ie/search?q=ROC+curve

## Viola-Jones Filter

https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework

http://www.vision.caltech.edu/html-files/EE148-2005-Spring/pprs/viola04ijcv.pdf









## Reinforcement Learning

https://en.wikipedia.org/wiki/Reinforcement_learning

## The Method of Temporal Differences

https://en.wikipedia.org/wiki/Temporal_difference_learning

## TD Gammon

https://en.wikipedia.org/wiki/TD-Gammon

## *The* RL Book

http://www.cs.ualberta.ca/~sutton/book/the-book.html

## Nice Dated RL Survey

https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/rl-survey.html

## Policy-Value Iteration

https://en.wikipedia.org/wiki/Markov_decision_process

## Q-Learning

https://en.wikipedia.org/wiki/Q-learning








Class Notes and Materials for Maynooth University CS401, Fall 2015
==================================================================

http://github.com/barak/mu-cs401-f2015

* Instructor: Prof Barak A. Pearlmutter
* Office: Computer Science, Ugly New Building room 132
* Class: Mon 10:00 Arts B; Tue 11:00 Arts C
* Office Hours: TBA

Administrative Matters
----------------------

* Scribes for notes.
* Markdown format.
* In git, on GitHub.
* Feel free to help with notes (fix typos as you read), send pull requests, etc.

Instructional Materials
-----------------------

* http://deeplearning.net/tutorial/
* https://github.com/hangtwenty/dive-into-machine-learning
* Curated list of Reinforcement Learning Resources: http://aikorea.org/awesome-rl/
* Curated list of Machine Learning Tutorials: https://github.com/ujjwalkarn/Machine-Learning-Tutorials/blob/master/README.md
