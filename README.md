## fullrmc
It's a Reverse Monte Carlo (RMC) python/Cython/C package, especially designed to solve an inverse problem whereby an atomic/molecular model is adjusted until its atoms positions have the greatest consistency with a set of experimental data.
RMC is probably best known for its applications in condensed matter physics and solid state chemistry.

fullrmc is a fully object-oriented package where everything can be overloaded allowing easy implementation and maintenance of the code. It's core sub-package and modules are fully optimized written in cython/C. 

fullrmc is unique in its approach, among other functionalities:

1. Atomic and molecular systems are supported.
2. All types (not limited to cubic) of periodic boundary conditions systems are supported.
3. Atoms can be grouped into groups so the system can evolve atomistically, clusterly, molecularly or any combination of those.
4. Every group can be assigned a different move generator.
5. Selection of groups to perform moves can be done manually OR automatically, randomly OR NOT !!


## Installation
fullrmc is still going through testing and further implementations. Amond others, the Core module is not uploaded to github yet
for protection purposes. A first version with full access to the code will be released soon only after the package gets published in a scientific journal.

## Online documentation
http://bachiraoun.github.io/fullrmc/

## Author
Bachir Aoun
