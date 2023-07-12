<h1> welcome to SNART: </h1>
a Synchrotron Self Absorption model for non-relativistic transients


****Provide link to the research note****

<h2> OVERVIEW: </h2>

<p><strong><span style="color: #ff0000;">SNART</span></strong> is a powerful and user-friendly tool that allows for the simultaneous fitting of multiple radio epochs and the estimation of physical parameters.<br /><strong><span style="color: #ff0000;">SNART</span></strong> is implemented in Python programming language and it uses the optimization library <a href="https://doi.org/10.5281/zenodo.11813">LMFIT</a> for least-squares minimization, performed in log-space.<br />Uncertainties are propagated from radio data observations, through the fit parameters, and to the physical parameters in the end-result using the <a href="http://pythonhosted.org/uncertainties/">Uncertainties Python package</a>. <br />This is the software behind <a href="https://ui.adsabs.harvard.edu/abs/2022ApJ...938...84D/abstract">DeMarchi+ 2022</a>, which serves as an example of its capabilities.<br />The SNART software package includes a set of functions for reading, manipulating and fitting radio data. The overview of the software is as follows:</p>
<ul>
<li>The user supplies radio observation data in a comma-separated values file, as well as a configuration file which contains details about the data, the transient event, and options for the fit.</li>
<li><strong><span style="color: #ff0000;">SNART</span></strong> parses the radio data and groups observations into SEDs according to a user-defined&nbsp; &Delta;t /t. <strong><span style="color: #ff0000;">SNART</span></strong> jointly fits these SEDs to a broken power-law, enforcing consistency in the model parameters &alpha;<sub>1</sub>, &alpha;<sub>2</sub>, and <em>s</em>.</li>
<li><strong><span style="color: #ff0000;">SNART</span></strong> uses fit results to calculate physical parameters <em>B, R, U,</em> n<sub>e</sub>, &rho;<sub>CSM</sub>, M<sup>&middot;</sup>/v<sub>wind</sub>, <em>p</em> (sometimes expressed as &gamma; in radio SN literature; we adopt <em>p </em>to avoid confusion with the electron Lorentz factor),&nbsp;<em>n</em> and&nbsp;<em>s</em> (the power-law evolution of the shocked and unshocked medium), <em>q</em> (the power-law evolution of <em>R(t)</em>), and the synchrotron cooling frequency.</li>
<li><strong><span style="color: #ff0000;">SNART</span></strong> produces outputs in the form of:
<ul>
<li>(i) text files containing the fit results and the derived physical parameters,</li>
<li>(ii) visualizations of the data, fitted SEDs, and physical parameters</li>

</ul>
</li>
</ul>
<p><br /><h2> HOW TO DOWNLOAD: </h2> <br />
  
-- Download the github repository
-- create a virtual environment
-- enter the directory while in your virtual environment
-- type `pip install -r requirements.txt` in the command line (without the ') to download versions of necessary packages
-- type `snart` to run the example


</p>
<p><br />&lt;h2&gt; Using the <a href="https://github.com/SNARTdev/SNART/tree/main/examples">Example</a> &lt;/h2&gt;</p>
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
