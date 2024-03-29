# Aperiodic Phased Arrays

---

#### Overview
BSc Project in Physics - Stochastic Optimisation of Aperiodic Phased Array Designs with Beamforming Techniques in the SAMI-2 Diagnostic.

This repository contains a module called `NonUniform_2D_PhasedArrayFunctions.py` for working with phased antenna arrays along with a jupyter notebook that takes you through examples of how to use various functions from the module. It has a collection of simple, and in some cases limited, performance evaluation functions. This code was created by Matthew Cox from the University of York as part of a Bachelor's project in Physics.



**Abstract to Paper:**

This project sought to determine the optimal arrangement of antenna elements on a phased antenna array. A phased array is an antenna system composed of multiple individual antenna elements arranged in a specific pattern. The focus was in using programs to automatically arrange antennas, score the layout and then iteratively improve their positions. In order to approximate array performance, novel numerical methods were created by the author. The overarching contribution made in this report is an improved phased antenna array design that will be integrated into the Synthetic Aperture Microwave Imager 2 (SAMI-2), a diagnostic that uses microwaves to probe various plasma edge properties inside the spherical tokamak in the MAST facility in Culham, UK. With regards to the methods and outcomes of the array design, the author
used stochastic optimisation in conjunction with beamforming techniques to iteratively improve the performance, on multiple metrics, of any given phased array. A variety of objective functions were tested as a proxy for the full beamformed derived performance. One such proxy objective function quantified the uniformity of the distribution of all antenna pair separations and angles; this showed promise in improving angular resolution, but to the detriment of sidelobe rejection. The array this paper proposes has 20 antenna elements, a predicted sidelobe rejection value of (−8.1 ± 0.2) dB, an angular resolution of (5.4 ± 0.1)◦ and a directivity of (13.4 ± 0.2) dB for microwaves at 20 GHz. The results of this study demonstrate the feasibility of using full beamforming calculations within an optimisation function when designing aperiodic arrays. As an addition to this project, images were reconstructed from virtual data using both the proposed array and the one currently in SAMI-2. However the validity of these images is greatly limited due to the assumption of idealised data.
