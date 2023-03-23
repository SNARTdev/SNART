<h1> welcome to SNART: </h1>
a Synchrotron Self Absorption model for non-relativistic transients


****Provide link to the research note****

<h2> OVERVIEW: </h2>

SNART is a powerful and user-friendly tool that allows for the simultaneous fitting of multiple radio epochs and the estimation of physical parameters.
SNART is implemented in Python programming language and it uses the optimization library \texttt{lmfit} for least-squares minimization, performed in log-space \citep{lmfit}.
Uncertainties are propagated from radio data observations, through the fit parameters, and to the physical parameters in the end-result using the Uncertainties Python package \cite{uncertainties}. 
This is the software behind DeMarchi+ 2022, which serves as an example of its capabilities.
The SNART software package includes a set of functions for reading, manipulating and fitting radio data. The overview of the software is as follows:.



<ul style="list-style-type:circle">
	<li>The user supplies radio observation data in a CSV file as well as a configuration text file containing details about the transient such as distance and explosion date and options for the fit.</li>
	<li>\snart reads radio data and groups individual SEDs according to a user-defined $\Delta t / t$. \snart jointly fits these SEDs to broken powerlaw, enforcing consistency in powerlaw slopes among all SEDs.</li>
  <li>\snart calculates physical parameters $B, R, U, n_e, \rho_{\rm{CSM}}$, $ \dot{M}/v_{\rm{wind}}$, $n$ and $s$ (the power law evolution of the shocked and unshocked medium), $q$ (the power law evolution of $R$ in time), and the frequency at which synchrotron cooling occurs.</li>
  <li>snart produces plots such as those shown in Figure \ref{fig:example} for each user-defined fit. Text files are written per model that contain the \texttt{lmfit} results. Dataframes containing the parameters of the previous step are saved to individual CSV files for each definition of $\epsilon$.</li>
</ul>


<h2> HOW TO DOWNLOAD: </h2>
<h2> DANIEL?: </h2>

Package version requirements:

pandas 0.24.2
uncertainties 3.1.2
lmfit 0.9.14
numpy 1.19.4
seaborn 0.11.2

scipy DANIEL???

python DANIEL????

- ********************** inputs in csv [WIP] ***********************

Name: name of supernova
Date: date of data point in MJD
Frequency: frequency of observation in GHz
Flux Density: Flux density in micro Janskys
Uncertainty: Uncertainty on flux density, also in micro Jy
Upper Limit: if the data point is an upper limit, write "*upper limit", otherise leave blank

- ********************** inputs in code ***********************

i=adiabatic index when using Ho 2019 definition of energy, otherwise a conversion factor to other microphysical
definitions of energy (5/3 = moatomic ideal gas. 1 = shuts this off (follows chev. 98))
s_value = guessing starting place for s, in the cases that hold s fixed, it will be at this value.
this parameter is fed to [[jointfitting.py](http://jointfitting.py/)] in the inputs to run_fit in the class SN_Fit
impostors = SEDs with datapoints that SHOULD be a BPL but the fit will think they're an SPL
theta = pitch angle
two_slope = when False, fits physical parameters (B, R, and onward) to a single power law evolution.
When set to True, automatically takes the median date and sets this as a break point for two slopes to fit
the slope of r vs t (for q). option to use custom midpoint with the `midpt` variable.
NOTE: manually set to FALSE for r vs. t, but left set == True for density vs. r
equipartition= TRUE or FALSE; for all cases at the same time

- ********************** output ***********************

in Physical_Params, a dataframe with the following columns:

SED: the number assigned to the group of points
date: date of the first point in the SED. [days since explosion]
nu pk: peak frequency in [5 Ghz]
flux pk: peak flux [Jy]
b: magnetic field [Gauss]
u: postshock energy corresponding to the energy density (2/(i+1))*rho_CSM v^2 [ergs]
r: radius of FS [cm]
limit: whether or not this is a limit on peak location (single-sided power law)
q: the PL of r vs t (unitless)
n_e: number density of electrons (per cm^3)
mv:	mdot over vwind. [v in units of 1000 km/s. mdot in units of 10^-4 * solar masses/year]
v: velocity of the forward shock in [cm/s] (NOT THE WIND)
den: [grams/cm^3]
coolingSync: frequency at which synchrotron cooling dominates. [Ghz]
s profile: PL of density vs r [unitless] (the PL density of the ambient medium)
integrated mass: integrating under the density vs radius curve. converted to solar masses.
n profile: PL of dense ejecta on the inside of the FS calculated from the s profile [unitless]
Unshocked, maintains information of outer layer of star.

the saved LMFIT outputs in text files within individual SEDs:

Will appear in untis of as [GHz] and [microJy] (this is how I keep them in my CSVs)
alpha1: the THIN slope of the SED
alpha2: the THICK slope of the SED

- ********************** known issues ***********************

The minimization doesn't do well at low fluxes/the display minimization is a different result from the observed minimization:
The minimization is run twice, once to do all calculations and again to plot the results. This is run twice to automatically
calculate and propagate uncertainties in a convenient manner. However, sometimes they are different from one another. This is
becuase the minimization takes place in logspace, and could do a better job if it was in linear space.

There are a ton of functions/commented out sections with the word "brem":
This alludes to Bremmsrahlung Emission, another capability of SNART that is in the works but has not been tested and
may not be compatible with the tweaks and changes I've made the the main body of code along the way. Think of this as
a half-step towards Bremmstrahlung implementation.


<h2> HOW TO CITE: </h2>
<h2> Here? my paper? </h2>
