# 2017 LANL Unified Host and Network Data Analysis Notebook

## Installation

Launch Jupyter Notebook or Lab:

$ jupyter notebook

$ jupyter lab

Oopen blog_winevents.ipynb using a Python 3 kernel. Recommend using Anaconda to have all of the packages installed. Additional packages beyond those in Anaconda:

$ conda install wget

$ conda install -c conda-forge umap-learn

Note: wget is only needed if you're downloading the data via the notebook, and incorporating umap logic in notebook is still a work in progress.


## Motivation

Microsoft Windows Event (WinEvent) logs provide a record of events occurring with the operating system, applications, accounts, and other system artifacts. Analyzing these logs for anomalous behavior (such as, unusual processes or account behavior) can be fruitful for cyber security purposes and have been used in this manner for some time (e.g., SANS). While there is a wealth of information within the WinEvent logs, there are also a number of challenges with analyzing the raw data — such as, not having appropriate features for separating or classifying the event or entity being logged. Context is often missing within the raw log entries and must be added back in for meaningful analysis and information. This notebook details the featuring engineering, principal component analysis (PCA), and K-means clustering steps that I took for classifying started processes logged (Event ID 4688) in the 2017 Los Alamos National Laboratory’s (LANL) dataset:

2017 LANL Unified Host and Network data set: https://csr.lanl.gov/data/2017.html

This work is motivated by: (i) the need for cyber security analytics to incorporate contextual information early into the process verse being joined in during manual triaging, and (ii) a concern that valuable latent features remain hidden or unused by security analytics. I explore the following questions:

* What are some features we can add into WinEvents and methods of engineering them?
* Can we reduce dimensionality and learn from feature weights of the principal components?
* What can we learn from clustering the results? Are there latent features that become visible in this example of process started WinEvents?


## File Descriptions

* blog_winevents.ipynb: the Jupyter notebook
* blog_winevents.html: an export of the notebook to html
* myLib.py: supporting functions for the notebook
* data directory: location of where the LANL data will be stored
* data/WinEventTypes.csv: csv file of WinEvent labels to add in


## Results

The results are detailed in a blog post on Medium:
https://medium.com/@mike_71681/exploring-latent-features-in-winevent-process-logs-113581564be4

I was able to engineer features that answer these questions:

* Is the event occurring during off-peak hours (“after hours”)
* Does the event coincide with system / Windows startup (“near startup”)
* Is the system locked and/or is the screensaver active
* What type of system recorded the event
* What type of user/account is tied to the event

Then reduce the dimensionality to 4 principal components using PCA, and interpret the meaning of each component.

Then classify the process started WinEvent records - clustering them into 5 clusters and interpreting their meaning:

* Cluster 0: an after hours process started from a user account
* Cluster 1: an after hours process started from a computer account
* Cluster 2: a peak hours process started from a user or computer account
* Cluster 3: an after hours process started from an application or service account when the system is locked or the screensaver is active
* Cluster 4: a peak hours process started when the system is locked or the screensaver is active

Clusters 1 & 2 contain almost 80% of the process started events, while 0, 3, and 4 are much more sparsely populated (particularly cluster 3).

## Acknowledgements

* Thank you to my coworker Jacob Baxter for his help and support when I got stuck, whether it is something annoying with Pandas or helping explain a concept.
* Thanks to the Udacity Data Science Nanodegree program for the hands-on lessons.


