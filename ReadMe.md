# Ashwinkumar
Introduction
The below code is to identify the 3 different species of Iris Flowers that exist, namely Setosa, Versicolor and Virginica. The species are differentiated based on 4 parameters. This code uses a K Nearest Neighbour Classification Algorithm to identify and classify the species.
Explanation of the dataset
The dataset used here is a standard, commonly available dataset called the iris dataset (GitHub link provided along with other references). This dataset contains 4 input parameters, namely the Sepal Length, Sepal Width, Petal Length and Petal Width (in that order). The last parameter is the output parameter, which is the species corresponding to that specific set of input parameters. In the GitHub link, these values are provided as a CSV file. However, the same dataset is also available in Scikit-learn, which is the Machine Learning Library I am using to train the model (explained later on in the file).
Machine Learning model used
The machine learning model used here is called the Nearest Neighbour Classification model. Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbours of each point: a query point is assigned the data class which has the most representatives within the nearest neighbours of the point. The k-neighbors classification in KNeighborsClassifier is the most commonly used technique. The optimal choice of the value 'k' is highly data-dependent: in general a larger k value suppresses the effects of noise, but makes the classification boundaries less distinct.
Implementation
Overall concept
This problem is relatively easy to implement using a free, widely used machine learning library for Python called Scikit-learn. This library does most of the model generation, training and testing, while still allowing us to modify certain values as required by the end user. In this case, as explained earlier, since the number of input parameters is low, and the input data is neatly labelled, this is a job best suited for a Nearest Neighbour Classsification Model. This is easily implemented using the 'KNeighboursClassifier' available in Scikit-learn.
Process
The entire Machine Learning process can be split into 7 distinct parts. They are listed below:
1.	Importing the Data
2.	Cleaning the Data
3.	Sorting the Data
4.	Splitting the Data
5.	Training the Model
6.	Analysing the performance of the Model
7.	Deployment of the model
These different parts are defined and elaborated below.
Part 1 and 2 - Importing and Cleaning the Data
To go about this project, we first need to import the data. This can either be done using the GitHub repo (link provided above), or through the in-built iris dataset in Scikit-learn. For this application, we shall use the in-built version, as it is easier to deal with than a CSV file. The next part of the process is to clean the data and remove points which are noisy. This need not be done in this case, as the dataset available in Scikit-learn is already verified, and considered to be clean. However, for other datasets, it is advisable to go through the data and clean it. Not doing this cleaning step can lead to the model either underfitting or overfitting.
A statistical model or a machine learning algorithm is said to have underfitting when a model is too simple to capture data complexities. It represents the inability of the model to learn the training data effectively result in poor performance both on the training and testing data.
A statistical model is said to be overfitted when the model does not make accurate predictions on testing data. When a model gets trained with so much data, it starts learning from the noise and inaccurate data entries in our data set. And when testing with test data results in High variance. Then the model does not categorize the data correctly, because of too many details and noise.
Part 3 - Sorting the Data
The third part of the process is to split the available data into a set of input and output parameters for the model to be trained. Since we are using a labelled dataset, Scikit-learn can easily split the data into input and output. This is necessary as the training function needs them to be separate. For other functions with may not need this split, this step can be skipped.
Part 4 - Splitting the Data
The fourth part of the process is to split the entire dataset into a training dataset, and a test dataset. This is a very important step, as it will determine how much data the model will learn from, hence determining the accuracy of the trained model.
If the test size is too low, we will have a lot of data used for training the model. However, this will lead to an underwhelming testing scenario, where we will not be able to test for multiple cases, hence not being able to completely verify the accuracy of the model.
If the test size is too high, then the training dataset will be small, which will make the model inaccurate, and be able to predict the answers accurately.
Hence, it is necessary to make sure that the test size is big enough to test for multiple cases, while still being small enough to have enough data for training the model. The industry standard for the test dataset size is around 20% and this is what we will also use. This dataset contains 150 entries in total, and hence we will end up with 120 rows of training data, and 30 rows of testing data.
Part 5 - Training the Model
The next part of the process is to train the model. As mentioned earlier, we will use the 'KNearestClassifier' for this. In this, need to provide the training data (both input and output), and the number of classifications we need the model to distinguish. In this case, we have 3 species of the iris flower, so the number of neighbours will be 3. This number can be changed based on the number of classifications we need our model to predict.
Part 6 - Analysing Model Performance
The sixth and final part of the model training is to use the test data, and predict the results. The predicted results can then be compared with the test data results, and an accuracy value can be determined. Here, the scikit-learn library also has functions to help with measuring accuracy, which is also used. This value ranges from 0 to 1, where a lower value represents the model being less accurate.
From here, the model can be tweaked to increase the accuracy. In general, an accuracy of above 90% is considered to be good. Here, we can see that the accuracy is 96.67%, which is better than expected. Hence, we can say that the model is well-trained, and very less to no further improvement is required.
Note that when the code is run multiple times, the accuracy will keep changing, which is primarily due to the entries which the model uses for training and testing. The split function used in step 4 is a random process, hence the actual data used in any given situation will not be the same, which might lead to small changes in the accuracy.
[5]:

Part 7 - Model Deployment
This is the final step of the Machine Learning process. Now that the model is trained and its accuracy is up to standard, we can use it to determine the results of a set of input data whose output is not known. In this case, we can make data up, based on the pre-existing iris dataset. For this example, let us make up 3 different cases.
1.	Case 1: Sepal Length = 4.4cm, Sepal Width = 4.4cm, Petal Length = 3cm, Petal Width = 1cm
2.	Case 2: Sepal Length = 5cm, Sepal Width = 3cm, Petal Length = 1cm, Petal Width = 2cm
3.	Case 3: Sepal Length = 4.8cm, Sepal Width = 3.4cm, Petal Length = 4cm, Petal Width = 0.5cm
Now, we can predict the species of the flower using the developed model.
Conclusions
Using this code, we are able to develop a model to identify 3 sub-species of the iris flower, and successfully train it. We used the Nearest Neighbours Classification Algorithm, provided by the Scikit-learn library. We were also able to determine its accuracy, and use it to make predictions about new and unseen data, for which we don't know the expected output.
References
1.	Python Version used (Python 3.11.5): https://www.python.org/downloads/release/python-3115/
2.	IDE used for creating the .py file (PyCharm):
3.	Jupyter Notebook (IDE used): https://jupyter.org/
4.	Iris Flower Dataset Explanation: https://en.wikipedia.org/wiki/Iris_flower_data_set
5.	Iris Flower dataset Github Link: https://gist.github.com/curran/a08a1080b88344b0c8a7
6.	Iris Flower dataset Scikit-learn Link: https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
7.	Scikit-learn Home Page: https://scikit-learn.org/stable/index.html
8.	K Nearest Neighbours Explanation (Geek for Geeks): https://www.geeksforgeeks.org/k-nearest-neighbours/
9.	Nearest Neighbours explanation (from Scikit-learn): https://scikit-learn.org/stable/modules/neighbors.html#classification
10.	KNeighborsClassifier Explanation (from Scikit-learn): https://scikit-learn.org/1.4/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
[ ]:
