#include <gdcpp.h>

struct Ackley
{
    static double pi()
    { return 3.141592653589; }

    Ackley()
    { }

    double operator()(const Eigen::VectorXd &xval, Eigen::VectorXd &) const
    {
        assert(xval.size() == 2);
        double x = xval(0);
        double y = xval(1);
        // Calculate ackley function, but no gradient. Let gradien be estimated
        // numerically.
        return -20.0 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + y * y))) -
            std::exp(0.5 * (std::cos(2 * pi() * x) + std::cos(2 * pi() * y))) +
            std::exp(1) + 20.0;
    }
};

int main()
{
    // Create optimizer object with Ackley functor as objective.
    //
    // You can specify a StepSize functor as template parameter.
    // There are ConstantStepSize, LimitedChangeStep, BarzilaiBorweinStep and
    // WolfeLineSearch available. (Default is BarzilaiBorweinStep)
    //
    // You can additionally specify a Callback functor as template parameter.
    //
    // You can additionally specify a FiniteDifferences functor as template
    // parameter. There are Forward-, Backward- and CentralDifferences
    // available. (Default is CentralDifferences)
    gdc::GradientDescent<double, Ackley,
        gdc::WolfeLineSearch<double, Ackley>> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaxIterations(200);

    // Set the minimum length of the gradient.
    // The optimizer stops minimizing if the gradient length falls below this
    // value (default is 1e-9).
    optimizer.setMinGradientLength(1e-6);

    // Set the minimum length of the step.
    // The optimizer stops minimizing if the step length falls below this
    // value (default is 1e-9).
    optimizer.setMinStepLength(1e-6);

    // Set the momentum rate used for the step calculation (default is 0.0).
    // Defines how much momentum is kept from previous iterations.
    optimizer.setMomentum(0.4);

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbosity(4);

    // Set initial guess.
    Eigen::VectorXd initialGuess(2);
    initialGuess << -2.7, 2.2;

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
