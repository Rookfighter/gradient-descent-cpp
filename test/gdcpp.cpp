/* gdcpp.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 12 Jul 2019
 */

#include "assert/eigen_require.h"
#include <gdcpp.h>

using namespace gdc;

template<typename Scalar>
struct Paraboloid
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    Scalar operator()(const Vector &state, Vector &)
    {
        return state(0) * state(0) + state(1) * state(1);
    }
};

template<typename Scalar>
struct Rosenbrock
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    Scalar operator()(const Vector &state, Vector &)
    {
        Scalar delta1 = 1 - state(0);
        Scalar delta2 = state(1) - state(0) * state(0);

        return delta1 * delta1 + 100 * delta2 * delta2;
    }
};

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;

TEST_CASE("gradient_descent")
{
    const float eps = 1e-3;

    SECTION("optimize paraboloid")
    {
        SECTION("forward differences")
        {
            GradientDescent<float,
                Paraboloid<float>,
                ConstantStepSize<float>,
                NoCallback<float>,
                ForwardDifferences<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("backward differences")
        {
            GradientDescent<float,
                Paraboloid<float>,
                ConstantStepSize<float>,
                NoCallback<float>,
                BackwardDifferences<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("central differences")
        {
            GradientDescent<float,
                Paraboloid<float>,
                ConstantStepSize<float>,
                NoCallback<float>,
                CentralDifferences<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("constant step size")
        {
            GradientDescent<float,
                Paraboloid<float>,
                ConstantStepSize<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("Barzilai-Borwein step")
        {
            GradientDescent<float,
                Paraboloid<float>,
                BarzilaiBorwein<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("Wolfe linesearch")
        {
            GradientDescent<float,
                Paraboloid<float>,
                WolfeBacktracking<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("Armijo linesearch")
        {
            GradientDescent<float,
                Paraboloid<float>,
                ArmijoBacktracking<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("Decrease linesearch")
        {
            GradientDescent<float,
                Paraboloid<float>,
                DecreaseBacktracking<float>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }
    }

    SECTION("optimize Rosenbrock")
    {
        GradientDescent<float, Rosenbrock<float>,
            WolfeBacktracking<float>> optimizer;
        optimizer.setMaxIterations(3000);
        optimizer.setMomentum(0.9);
        Vector xval(2);
        xval << -0.5, 0.5;
        Vector xvalExp(2);
        xvalExp << 1, 1;

        auto result = optimizer.minimize(xval);
        REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
    }
}
