=============================================================

Clustering with k-means
Comparison with hierarchical clustering
PCA
=============================================================

Purdue University Spring 2017 CS 573 Data Mining Homework 5
=============================================================

Author: Parag Guruji
Email: pguruji@purdue.edu
=============================================================
 
Python Version 2.7
=============================================================

Directory structure:

parag_guruji/
	|---hw5.py
	|---analysis.pdf
	|---README.txt
	|---requirements.txt
	|---output/ (optional)
	|	|---exploration1
	|	|	|---0.png
	|	|	|---1.png
	|	|	|---2.png
	|	|	|---3.png
	|	|	|---4.png
	|	|	|---5.png
	|	|	|---6.png
	|	|	|---7.png
	|	|	|---8.png
	|	|	|---9.png
	|	|---exploration2
	|	|	|---digit-clusters.png
	|	|---analysis
	|	|	|---B1.csv
	|	|	|---B3.csv
	|	|	|---B3_mean.csv
	|	|	|---B3_var.csv
	|	|	|---B1SC.png
	|	|	|---B1WC.png
	|	|	|---B4ClusterData1K8.png
	|	|	|---B4ClusterData2K4.png
	|	|	|---B4ClusterData3K2.png
	|	|	|---B4NMI.png
	|	|---comparison
	|	|	|---C3.csv
	|	|	|---C1DendogramSingleLinkage.png
	|	|	|---C2DendogramCompleteLinkage.png
	|	|	|---C3DendogramAverageLinkage.png
	|	|	|---C3SC.png
	|	|	|---C3WC.png
	|	|	|---C5NMI.png
	|	|	|---C5ImageLabels.png
	|	|	|---C5ZSingle.txt
	|	|	|---C5ZComplete.txt
	|	|	|---C5ZAverage.txt
	|	|---bonus
	|	|	|---Bonus3ClustersByPCA.png
	|	|	|---eigen_1.png
	|	|	|---eigen_2.png
	|	|	|---eigen_3.png
	|	|	|---eigen_4.png
	|	|	|---eigen_5.png
	|	|	|---eigen_6.png
	|	|	|---eigen_7.png
	|	|	|---eigen_8.png
	|	|	|---eigen_9.png
	|	|	|---eigen_10.png
	|	|	|---analysis
	|	|	|	|---B1.csv
	|	|	|	|---B1Data1SC.png
	|	|	|	|---B1Data1WC.png
	|	|	|	|---B1Data2SC.png
	|	|	|	|---B1Data2WC.png
	|	|	|	|---B1Data3SC.png
	|	|	|	|---B1Data3WC.png
	|	|	|	|---B4ClusterData1K8.png
	|	|	|	|---B4ClusterData2K4.png
	|	|	|	|---B4ClusterData3K2.png
	|	|	|	|---B4NMI.png


=============================================================

usage: hw5.py [-h] [-r rawDigitsFilename] [-x exploration] [-a analysis]
              [-c comparison] [-p]
              dataFilename kValue

CS 573 Data Mining HW5 Clustering

positional arguments:
  dataFilename          file-path of embeddings input data
  kValue                Number of clusters to use

optional arguments:
  -h, --help            show this help message and exit
  -r rawDigitsFilename, --rawDigitsFilename rawDigitsFilename
                        file-path of raw digits input data (default: None)
  -x exploration, --exploration exploration
                        Run exploration question 1 or 2 as chosen. (default:
                        None)
  -a analysis, --analysis analysis
                        Solutions to Part B - Analysis of k-means: 1. Question
                        B.1 2. Question B.3 3. Question B.4 (default: None)
  -c comparison, --comparison comparison
                        Solutions to Part C - Comparison to hierarchical
                        clustering: 1. Question C.1, C.2 and C.3 2. Question
                        C.5 (default: None)
  -p, --pca             Solutions to Bonus - PCA (default: False)
