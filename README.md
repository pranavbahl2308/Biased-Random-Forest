# Biased-Random-Forest

The data imbalance problem has existed since we have possibly started storing the data. Most of the techniques used in machine learning to overcome the data imbalance problem includes oversampling or undersampling  the data. But Biased Random Forest(BRAF) algorithm moves the concept of oversampling from the data level to the algorithm level. In this approach the algorithm combines two random forest algorithm; in one it uses the input data and in the other it resamples the provided data by removing the samples from majority class label that are not close to any of the minority class samples. The closeness of the samples is calculated using a distance measure; in my implementation I have used Euclidean distance.
        This is an attempt for implementing a version of Random Forest algorithm proposed at : https://ieeexplore.ieee.org/document/8541100 to overcome class imbalance problem for classification. I have implemented the BRAF algorithm in Python. For this project I have not used existing python packages with Random Forest or K Nearest Neighbors implementation instead I am using the existing Random Forest implementation available at https://github.com/SebastianMantey/Random-Forest-from-Scratch. Also, I have provided the reference for each code block that I have not implemented myself.
    
    
    Future work
    
    Some of the things that can be done to enhance my implementation of BRAF includes:
    - Move codebase from notebook to python script
    - include data checks to avoid unexpected failures
    - write tests to validate the expected functionality of codebase
    - further segment code into classes for better packaging
    - avoid data pressumptions, such as binary labels and last feature in the dataset being the class label
    - add other distance measures for creating critical dataset
    - code optimizations to decrease the run time
