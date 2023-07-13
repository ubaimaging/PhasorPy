
# PhasorPy: A Python library for phasor analysis

The significance of time-resolved (FLIM) and hyperspectral imaging (HSI) in biomedical science cannot be overstated. Traditional imaging and spectroscopy fusion have unlocked new and previously inaccessible data. The Phasor plots have emerged as a revolutionary tool for bioimaging analysis due to their straightforward approach. Consequently, it has been instrumental in democratizing access to FLIM and HSI, making it a crucial player in the field.


The PhasorPy library is built upon SimFCS, a program created by Enrico Gratton at the Laboratory for Fluorescence Dynamics at the University of California, Irvine. This library is designed for FLIM and HSI data analysis, utilizing the phasor technique, which is a model-free method that relies on the Fourier transform properties.



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

The angular position in the spectral phasor plot represents the emission spectrum's center of mass, while the spectrum's full width determines the modulus at half maximum (FWHM). If the spectrum is broad, its location should be near the center. However, if there is a red shift in the spectrum, its location will move counterclockwise towards the increasing angle from position (1, 0). The properties of spectral phasor plots are similar to those of lifetime phasors. To learn more about the specifics of spectral phasor plot properties, refer to Malacrida et al. 1.


## Installation

```bash
  pip install PhasorPy
  conda install PhasorPy
```
    
## Demo

### Phasor and Pseudocolor representation

Obtain the phasor plot. From the average intensity image users can obtain 
the cutoff intensity in order to remove the background.  

Its also allows users to get the pseudocolor RGB image from the phasor, 
using three components.

![fig1](https://github.com/bschuty/PhasorPy/blob/main/Figures/Figure_1.png)

### Phasor plot

This funtionality allows users to obtain one or many phasors in the same plot. 

![fig2](https://github.com/bschuty/PhasorPy/blob/main/Figures/Figure_2.png)

### Phasor components determination

To obtain the component percentage between two components and visualize its histogram. 

![fig2](https://github.com/bschuty/PhasorPy/blob/main/Figures/Figure_3.png)





## Authors

- [@schutyb](https://github.com/schutyb)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Contributing

We welcome all contributions to the PhasorPy library. We aim to create a collaborative and open-source community that develops spectroscopy and fluorescence microscopy analysis tools. We aim to promote self-sustainability in the long term, similar to other Python libraries and communities while ensuring broad access to these tools.


## References

[1] Malacrida, L., Gratton, E. & Jameson, D. M. Model-free methods to study 
membrane environmental probes: A comparison of the spectral phasor and 
generalized polarization approaches. Methods Appl. Fluoresc. 3, 047001 (2015).
## Used By

This project is used and maintain by:

- Advanced Bioimaging Unit is a joint initiative between the Institut Pasteur de Montevideo and Universidad de la República, Uruguay.

