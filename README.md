This repository contains the source code of the following paper:

Imani, E., Radkani, S., Hashemi, A., Harati, A., Pourreza, H., & Goudarzi, M. M. (2022). Distributed coding of evidence accumulation 
across the mouse brain using microcircuits with a diversity of timescales. bioRxiv, 2022-09.

pipeline of the scripts:
dPCA analysis:
1) run averageRate.m to compute average firing rate
2) run featureExtraction.m to prepare firing rate matrix for dPCA analysis
3) run dpca_EI.m for dPCA and clustering analyses
4) run dpca_auroc.m, plotProjection.m, and plot_dpca_clusters.m for the visualization

Find DDM-like neurons:
1) run TemporalAURoc.m, TemporalAURoc_chance.m, and computeSigAuroc.m to compute choice probability and evidence selectivity
2) run averageRate.m to compute average firing rate
3) run finalSelectiveUnits_clusteringRes, prepareFR_visualGrading.m, and visualGradingSelectiveUnits.m to find DDM-like neurons

Accumulator analysis:
1) run createDataTotalUnits.m to generate subpopulations
2) run parallel_fitting_single_collapsBound.py for single accumulator model
3) run parallel_fitting_race_collapsBound.py for independent race accumulator model
4) run parallel_fitting_collapsBound.py for dependent race accumulator model
5) run EV_rSquaredBestParam.py to compute log-likelihood and R-squared values
6) run selectBestFits.py for greedy search across different parameters
7) run timescaleData.py for sampling spikes from the trained models
8) run analyze_ddm_fitting.m for model comparison, timescale computation, visualization, ....


