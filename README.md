## Baselines
In this section, all the baselines along with their proper mathematical formulation are given. The baseline algorithms used are:

 1. Modified Random Sample Consensus(RANSAC)
 2. Sample Rate Compression (SRC)
 3. K-Means Clustering


We assume the number of input datapoints $$|D_N| = N$$ and required number of down-sampled datapoints $$D_M$$ = $$M$$

### Random Sample Consensus (RANSAC)
RANSAC [] is a model-fitting method commonly used in a lot of Computer Vision applications. In this case, the data-points are $(X, Y)$ location co-ordinates. Initially, two points are randomly selected and "inliers" are obtained. "Inliers" are the location co-ordinates which lie along the line connecting the two points. Thus the output of RANSAC is all the lines in the data-set. 

In Bayes-Swarm, the robot movements are straight lines. Hence, we use RANSAC to extract all the lines (trajectories)(for instance $k$). To sample $M$ co-ordinates, we sample $M/k$ points from each line/trajectory. This is illustrated in the pseudocode given in Algorithm \ref{alg:modifiedransac} 

**Modified RANSAC Algorithm** <br/>
Given: Input data $$D$$, Required points count: $$M$$
Threshold: $$H$$
$$D_M \gets$$ Down-sampled Dataset

While  $$|D |> 0 :$$ <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    $$S_i =  RANSAC(D) $$  <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    $$S.append(S_i)$$ <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    Remove $$S_i$$ from $$D$$ <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    If $$length(D) < H$$  <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;break <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    EndIf    <br/>
EndWhile <br/>

$$k = length(S)$$	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Calculate the number of lists in  S <br/>

$$P_{\text{per-list}} = \frac{M}{k}$$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;#Calculate the required points per list <br/>

**For** $i = 1$ **to** $N$ <br/>
 &nbsp;&nbsp;&nbsp;    $$Q \gets P_{\text{per-list}}$$ Equidistant points from $$S[i]$$ <br/>
&nbsp;&nbsp;&nbsp;    $$D_M$$.append($$Q$$) <br/>
**EndFor** <br/>

return  $$D_M$$ <br/>

### Sample Rate Compression (SRC)
Sample Rate Compression[] is used a lot in signal processing domain. After sampling the first point, every $k^{th}$ point is sampled where $k = \frac{N}{M}$. SRC was used in the Bayes-Swarm algorithm proposed by Ghassemi et al [], []. 

### K-Means Clustering
Clustering has been used in literature to down-sample data in Machine Learning [], as well as for Gaussian Processes(GP)[] As per Liu et al [], the data is clustered in $M$ clusters and the $M$ cluster heads as the down-sampled dataset. Here we used K-Means Clustering to cluster the data.



## Numerical Experiments
In this section, we provide an analysis of the performance of our method against baselines for the parameters affecting the number and nature of observations. The parameters that we have considered are number of robots, robot speed and sensor sampling rate. In the paper we have shown experiments with a swarm of  35  robots, with the speed of 1 m/s and sensor sampling rate of 1 Hz. Here, we will show the results for a swarm of 20 robots and 50 robots, with speeds of 0.2 m/s and 1 m/s and sensor sampling rate of 5 Hz along with 1 Hz. 

All the experiments were performed on the environment shown Figure 5(a) with highly Multi-modal signal. <br/>

#### Environment 1
**Experiments with robot-speed = 1m/s and sensor-sampling-rate=1Hz** <br/>
<img src="RAL_results/results_plots/png_plots/env1_results/env1_20_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env1_results/env1_35_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env1_results/env1_50_robots_speed1.png" width="200" height="200">

**Experiments with robot-speed = 0.2m/s and sensor-sampling-rate=1Hz** <br/>
<img src="RAL_results/results_plots/png_plots/env1_results/without_ransac/env1_20_robots_speed_point2.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env1_results/without_ransac/env1_35_robots_speed_point2.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env1_results/without_ransac/env1_50_robots_speed_point2.png" width="200" height="200">

**Experiments with robot-speed = 1m/s and sensor-sampling-rate=5Hz** <br/>
<img src="RAL_results/results_plots/png_plots/env1_results/freq_5/env1_20_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env1_results/freq_5/env1_35_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env1_results/freq_5/env1_50_robots_speed1.png" width="200" height="200">


#### Environment 2
**Experiments with robot-speed = 1m/s and sensor-sampling-rate=1Hz** <br/>
<img src="RAL_results/results_plots/png_plots/env2_results/env2_20_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env2_results/env2_35_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env2_results/env2_50_robots_speed1.png" width="200" height="200">

**Experiments with robot-speed = 1m/s and sensor-sampling-rate=5Hz** <br/>
<img src="RAL_results/results_plots/png_plots/env2_results/freq_5/env2_20_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env2_results/freq_5/env2_35_robots_speed1.png" width="200" height="200">
<img src="RAL_results/results_plots/png_plots/env2_results/freq_5/env2_50_robots_speed1.png" width="200" height="200">

