# Baselines
In this section, all the baselines along with their proper mathematical formulation are given. The baseline algorithms used are:

 1. Modified Random Sample Consensus(RANSAC)
 2. Sample Rate Compression (SRC)
 3. K-Means Clustering

We assume the number of input datapoints $|D_N| = N$ and required number of down-sampled datapoints = $M$

\subsection{Random Sample Consensus (RANSAC)}
RANSAC [] is a model-fitting method commonly used in a lot of Computer Vision applications. In this case, the data-points are $(X, Y)$ location co-ordinates. Initially, two points are randomly selected and "inliers" are obtained. "Inliers" are the location co-ordinates which lie along the line connecting the two points. Thus the output of RANSAC is all the lines in the data-set. 

In Bayes-Swarm, the robot movements are straight lines. Hence, we use RANSAC to extract all the lines (trajectories)(for instance $k$). To sample $M$ co-ordinates, we sample $M/k$ points from each line/trajectory. This is illustrated in the pseudocode given in Algorithm \ref{alg:modifiedransac} 

Modified RANSAC Algorithm
Given: Input data $D$, Required points count: $M$
Threshold: $H$
$D_M \gets$ Down-sampled Dataset

While  $|D |> 0 :$
&nbsp;&nbsp;    $S_i =  RANSAC(D) $  
&nbsp;&nbsp;    $S.append(S_i)$
&nbsp;&nbsp;    Remove $S_i$ from $D$
&nbsp;&nbsp;    If $length(D) < H$ 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;break
&nbsp;&nbsp;    EndIf    
EndWhile

$k = length(S)$	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Calculate the number of lists in  S

$P_{\text{per\_list}} = \frac{M}{k}$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Calculate the required points per list

For $i = 1$ to $N$
 &nbsp;&nbsp;&nbsp;    $Q$ $\gets$ $P_{\text{per\_list}}$ Equidistant points from $S[i]$
&nbsp;&nbsp;&nbsp;    $D_M$.append($Q$)
\EndFor

return  $D_M$

## Sample Rate Compression (SRC)
Sample Rate Compression[] is used a lot in signal processing domain. After sampling the first point, every $k^{th}$ point is sampled where $k = \frac{N}{M}$. SRC was used in the Bayes-Swarm algorithm proposed by Ghassemi et al [], []. 

## K-Means Clustering
Clustering has been used in literature to down-sample data in Machine Learning [], as well as for Gaussian Processes(GP)[] As per Liu et al [], the data is clustered in $M$ clusters and the $M$ cluster heads as the down-sampled dataset. Here we used K-Means Clustering to cluster the data.
