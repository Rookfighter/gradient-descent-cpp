#include <gdcpp.h>

// Implement an objective functor.
struct Paraboloid
{
    float operator()(const Eigen::VectorXf &xval, Eigen::VectorXf &gradient) const
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
    // You can additionally specify a Callback functor as template parameter.
    // You can additionally specify a FiniteDifferences functor as template
    // parameter. Default is CentralDifferences. Forward- and BackwardDifferences
    // are also available.
    gdc::GradientDescent<float, Paraboloid> optimizer;

    // Set number of iterations as stop criterion.
    // Set it to 0 or negative for infinite iterations (default is 0).
    optimizer.setMaxIterations(100);

    // Set the minimum length of the gradient.
    // The optimizer stops minimizing if the gradient length falls below this
    // value (default is 1e-6).
    optimizer.setMinGradientLength(1e-3f);

    // Set the learning rate used for the step calculation (default is 0.7).
    optimizer.setLearningRate(0.8f);

    // Set the momentum rate used for the step calculation (default is 0.9).
    // Defines how much momentum is kept from previous iterations.
    optimizer.setMomentumRate(0.8f);

    // Turn verbosity on, so the optimizer prints status updates after each
    // iteration.
    optimizer.setVerbose(true);

    // set initial guess
    Eigen::VectorXf initialGuess(2);
    initialGuess << 2, 2;

    // start the optimization
    auto result = optimizer.minimize(initialGuess);

    std::cout << "Done! Converged: " << (result.converged ? "true" : "false")
        << " Iterations: " << result.iterations << std::endl;

    // do something with final function value
    std::cout << "Final fval: " << result.fval << std::endl;

    // do something with final function value
    std::cout << "Final xval: " << result.xval.transpose() << std::endl;

    return 0;
}
