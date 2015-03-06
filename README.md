## fullrmc
It's a Reverse Monte Carlo (RMC) python/Cython/C package, especially designed to solve an inverse 
problem whereby an atomic/molecular model is adjusted until its atoms positions have the greatest 
consistency with a set of experimental data. RMC is probably best known for its applications in 
condensed matter physics and solid state chemistry. fullrmc is a fully object-oriented package 
where everything can be overloaded allowing easy development, implementation and maintenance of the code. 
It's core sub-package and modules are fully optimized written in cython/C. fullrmc is unique in its approach, 
among other functionalities:

1. Atomic and molecular systems are supported.
2. All types (not limited to cubic) of periodic boundary conditions systems are supported.
3. Atoms can be grouped into groups so the system can evolve atomically, clusterly, molecularly or any combination of those.
4. Every group can be assigned a different move generator (translation, rotation, a combination of moves generators, etc).
5. Selection of groups to perform moves can be done manually OR automatically, randomly OR NOT !!

## Next on the list
* Creating additional agitation move generators such as torsionAngle, etc.
* Creating additional translation move generators such as translation along axis defined between two groups or random positions, etc.
* Elements transmutation and atoms switching.

## Installation
fullrmc is still going through testing and further implementations. 
Among others, the Core module is not yet uploaded to github for protection purposes. 
A first version with full access to the code will be released soon only after the package gets officially published.

## Online documentation
http://bachiraoun.github.io/fullrmc/

## Author
Bachir Aoun

## Copyright and license
No license is provided, therefore the code is copyrighted by default. 
People can read the code, but they have no legal right to use it. To use the code, 
you must contact the author directly and ask permission.
