
# PhasorPy: A Python library for phasor analysis

Time-resolved (FLIM) and hyperspectral imaging (HSI) have become paramount 
in biomedical science. The power of the combination between traditional 
imaging and spectroscopy opens the possibility to address information 
inaccessible before. For bioimaging analysis of these data, the Phasor 
plots are a tool revolutionizing the field because of their straightforward 
approach. Thus it is becoming a key player in democratizing access to FLIM and HSI


PhasorPy library is based on SimFCS, a software developed 
by Enrico Gratton at the Laboratory for Fluorescence Dynamic,
University of California, Irvine. PhasorPy is a library for FLIM and HSI data analysis 
using the phasor approach. The phasor approach was developed as model free method 
and relies on the fourier transform properties.



## Documentation


### Phasor Analysis 
Considering an hyperspectral image, the fluorescence spectra at each pixel can be
transformed in phasor coordinates (G (λ)) and (S (λ)) as described in the following 
equations. I(λ) represent the intensity at every wavelength (channel), n is the 
number of the harmonic and λ i the initial wavelength. The, x and y coordinates 
are plotted in the spectral phasor plot.

![eq1](https://github.com/bschuty/PhasorPy/blob/main/Figures/equation_spectral.png)

The position for every pixel in the spectral phasor plot can be defined by the phase
angle and the modulus (M) given the coordinates G and S.

![eq2](https://github.com/bschuty/PhasorPy/blob/main/Figures/equation_spectral_mp.png)

The angular position in the spectral phasor plot relates to the center of mass of 
the emission spectrum and the modulus depends on the spectrum’s full width at 
the half maximum (FWHM). For instance, if the spectrum is broad its location 
should be close to the center. Otherwise, if there is a red shift in the spectrum,
its location will move counterclockwise toward increasing angle from position
(1, 0). Spectral phasors have the same vector properties as lifetime phasors. 
A detailed description of the spectral phasor plot properties can be found in 
Malacrida et al. 1. 


## Installation

```bash
  pip install PhasorPy
  conda install PhasorPy
```
    
## Demo

### Phasor and Pseudocolor representation

Obtain the phasor plot. From the average intensity image users can obtain 
the cutoff intensity in orther to clean up the background. Having a much cleaer 
phasor representation. 

Its also allows users to get the pseudocolor RGB image rom the phasor, 
with three components.

![fig1](https://github.com/bschuty/PhasorPy/blob/main/Figures/Figure_1.png)

### Phasor plot

This funtionality allows users to obtain one or many phasors in the same plot. 

![fig2](https://github.com/bschuty/PhasorPy/blob/main/Figures/Figure_2.png)

### Phasor components determination

To obtain the component percentage between two components and visualize its histogram. 

![fig2](https://github.com/bschuty/PhasorPy/blob/main/Figures/Figure_3.png)





## Authors

- [@bschuty](https://www.github.com/bschuty)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Contributing

Contributions are always very well welcome. The PhasorPy library intends 
to create an open-source and collaborative community between spectroscopy 
and fluorescence microscopy users with the same functionalities as SimFCS 
but accessible and self-sustainable in the long term as other Python 
libraries and communities. 


## References

[1] Malacrida, L., Gratton, E. & Jameson, D. M. Model-free methods to study 
membrane environmental probes: A comparison of the spectral phasor and 
generalized polarization approaches. Methods Appl. Fluoresc. 3, 047001 (2015).
## Used By

This project is used and maintain by:

- Advance Bioimaging Unit at Institt Pasteur Montevideo and 
  Universidad de la República

