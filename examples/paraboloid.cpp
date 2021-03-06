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
    // There are ConstantStepSize, BarzilaiBorwein and
    // WolfeBacktracking available. (Default is BarzilaiBorwein)
    //
    // You can additionally specify a Callback functor as template parameter.
    //
    // You can additionally specify a FiniteDifferences functor as template
    // parameter. There are Forward-, Backward- and CentralDifferences
    // available. (Default is CentralDifferences)
    gdc::GradientDescent<double, Paraboloid,
        gdc::ConstantStepSize<double>> optimizer;

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
    optimizer.setStepSize(gdc::ConstantStepSize<double>(0.8));

    // Set the momentum rate used for the step calculation (default is 0.0).
    // Defines how much momentum is kept from previous iterations.
    optimizer.setMomentum(0.1);

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbosity(4);

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
