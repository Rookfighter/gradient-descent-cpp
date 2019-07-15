# gradient-descent-cpp

![Cpp11](https://img.shields.io/badge/C%2B%2B-11-blue.svg)
![License](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Travis Status](https://travis-ci.org/Rookfighter/gradient-descent-cpp.svg?branch=master)
![Appveyor Status](https://ci.appveyor.com/api/projects/status/66uh2rua4sijj4y9?svg=true)

gradient-descent-cpp is a header-only C++ library for gradient descent
optimization using the Eigen3 library.

## Install

Simply copy the header file into your project or install it using
the CMake build system by typing

```bash
cd path/to/repo
mkdir build
cd build
cmake ..
make install
```

The library requires Eigen3 to be installed on your system.
In Debian based systems you can simply type

```bash
apt-get install libeigen3-dev
```

Make sure ```Eigen3``` can be found by your build system.

## Usage

There are three steps to use gradient-descent-cpp:

* Implement your objective function as functor
* Instantiate the gradient descent optimizer
* Choose your parameters

```cpp
#include <gdcpp.h>

// Implement an objective functor.
struct Paraboloid
{
    double operator()(const Eigen::VectorXd &xval, Eigen::VectorXd &gradient)
    {
        // compute gradient explicitly
        // if gradient calculation is omitted, then the optimizer uses
        // finite differences to approximate the gradient numerically
        gradient.resize(2);
        gradient(0) = 2 * xval(0);
        gradient(1) = 2 * xval(1);

        return xval(0) * xval(0) + xval(1) * xval(1);
    }
};

int main()
{
    // Create optimizer object with Paraboloid functor as objective.
    //
    // You can specify a StepSize functor as template parameter.
    // There are ConstantStepSize, BarzilaiBorweinStep and WolfeLineSearch
    // available. (Default is WolfeLineSearch)
    //
    // You can additionally specify a Callback functor as template parameter.
    //
    // You can additionally specify a FiniteDifferences functor as template
    // parameter. There are Forward-, Backward- and CentralDifferences
    // available. (Default is CentralDifferences)
    gdc::GradientDescent<double, Paraboloid,
        gdc::ConstantStepSize<double, Paraboloid>> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaxIterations(100);

    // Set the minimum length of the gradient.
    // The optimizer stops minimizing if the gradient length falls below this
    // value (default is 1e-9).
    optimizer.setMinGradientLength(1e-6);

    // Set the minimum length of the step.
    // The optimizer stops minimizing if the step length falls below this
    // value (default is 1e-9).
    optimizer.setMinStepLength(1e-6);

    // Set the the parametrized StepSize functor used for the step calculation.
    optimizer.setStepSize(gdc::ConstantStepSize<double, Paraboloid>(0.8));

    // Set the momentum rate used for the step calculation (default is 0.9).
    // Defines how much momentum is kept from previous iterations.
    optimizer.setMomentum(0.8);

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbose(true);

    // Set initial guess.
    Eigen::VectorXd initialGuess(2);
    initialGuess << 2, 2;

    // Start the optimization
    auto result = optimizer.minimize(initialGuess);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false")
        << " Iterations: " << result.iterations << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval << std::endl;

    // do something with final x-value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;

    return 0;
}
```
