# Documentation
Documentation is added at sufficient milestones.

## October 3rd, 2018
With the new and improved dataset I am getting an extremely high level of performance out of my algorithm. It is
succeeding very well at what it does and has exhibited extremely high accuracy in both cases across multiple system
runs.

Training Accuracy: 99%
Test Accuracy: 99%

## September 12th, 2018
The support vector machine implementation via tensorflow allowed for a very marginal increase in
training time for the algorithm, and a very negligable increase in overall predictive accuracy of
the dataset. I feel that I have reached the end of the usefulness of this dataset, but I am now
moving onto the more robust NHANES dataset to be able to make much more real life estimates about
the outcome and lifestyle choices that affect diabetes in adults.

I need to do more research to determine the viability of the insulin prediction system to know more
concretely whether or not it is feasible given my current resources.

## August 29th, 2018
The support vector machine performed considerably better with the Pima Indians Data Set
Training Accuracy: 83%
Test Accuracy: 78%

Right now the results are exhibiting signs of bias which will equire additional tuning to fix.
I also would ideally want to have a training set and test set accuracy of ~90% at least before I move
onto the next phase of the research project.

Adjustment of the inverse regularization parameter, C, moves the training and test accuracies into a similar range
with each being 78% ish respectively. So far hyperparameter tuning has proved to not change the ouput
much besides that. I will inspect the data and see how certain columns affect this data further.

## August 25th, 2018
Random Forest Regressor Stats on the Pima Indians Dataset
Percent Successful Prediction: 69%

The Random Forest Regressor did not perform well. I will be doing additional exploration into
different learning algorithms to find a more performant option.

## Citations
### WebMD
#### How alcohol affects diabetes
https://www.webmd.com/diabetes/guide/drinking-alcohol

### NIDDK - National Institude of Diabetes and Digestive and Kidney Diseases
https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke

### Pima Indians Database
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
The purposes of it for my research are to use it as a base training point as well as springboard for further
experimentation with my algorithms.

### Sebastian Raschka
I used some code samples as shown in his "Python Machine Learning, Second Edition" book for a few small data
visualization things. Most code is mine however small parts of functions contain code that came from this book.
