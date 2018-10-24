# Lazarus: Diabetes Classification via High-Performance Classifiers
Lazarus of Bethany, also known as Saint Lazarus or Lazarus of the Four Days, is the subject of a prominent miracle of Jesus in the Gospel of John, in which Jesus restores him to life four days after his death. This project has no religious affiliation, but the idea of the tale of Lazarus is a very interesting one and a neat concept to think about for my medical research. As long as we don't create a zombie apocalype, that is.

## Getting Started With This Classifier
These instructions will get you a copy of the classifier on your local machine for use with your own data (see below for proper measurement criteria for patients). Inside of the `core` directory will be tests on sample data to verify functionality as well as implementations for the given research here.

### Prerequisites
Currently, this has only been tested on Linux 4.18.5+ and has not been tested on other platforms at this time
- CMake >= 3.9
- C++ Compiler
- STDC++17

#### Arch Linux
```
pacman -S cmake clang
```

#### Installing
The project can be cloned anywhere on the desired system. PROJECT\_ROOT is assumed to be in that place.

CMake supports out of source builds, so we can make a build directory (refered herein as `BUILD_DIR`).
```
cd <PROJECT_ROOT>
mkdir <BUILD_DIR>
```

After the place to build our system has been created we can now use CMake to generate our build structure.
```
cd <BUILD_DIR>
cmake ..
```

By default, CMake defaults the install path of `make install` to /usr/local on UNIX. To install elsewhere you can add `DCMAKE_INSTALL_PREFIX` onto the cmake command.
```
cmake .. -DCMAKE_INSTALL_PREFIX=./install
```

Once all build files are prepared, we can compile.
```
make
```

And then install.
```
make install
```

#### Uninstalling
To uninstall simply execute `$ rm -r <PROJECT_ROOT>` and all files should be fully removed from the computer then. Note, if you installed to a different folder, that will remain present upon deletion of the project root and will also need to be removed.

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](https://github.com/jparr721/Lazarus/blob/master/LICENSE) file for details.

## Acknowledgements
- Grand Valley State University DEN Research Lab members.
- Dr. Greg Wolffe
- Roberto Sanchez
- Lawrence O'Boyle
- Any of the numberous guides and walkthroughs I have followed thus far.
