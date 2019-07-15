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
                ConstantStepSize<float, Paraboloid<float>>,
                NoCallback<float>,
                ForwardDifferences<float, Paraboloid<float>>> optimizer;
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
                ConstantStepSize<float, Paraboloid<float>>,
                NoCallback<float>,
                BackwardDifferences<float, Paraboloid<float>>> optimizer;
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
                ConstantStepSize<float, Paraboloid<float>>,
                NoCallback<float>,
                CentralDifferences<float, Paraboloid<float>>> optimizer;
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
                ConstantStepSize<float, Paraboloid<float>>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("barzilai borwein step")
        {
            GradientDescent<float,
                Paraboloid<float>,
                BarzilaiBorweinStep<float, Paraboloid<float>>> optimizer;
            optimizer.setMaxIterations(100);

            Vector xval(2);
            xval << 2, 2;
            Vector xvalExp(2);
            xvalExp << 0, 0;

            auto result = optimizer.minimize(xval);
            REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
        }

        SECTION("Wolfe line search")
        {
            GradientDescent<float,
                Paraboloid<float>,
                WolfeLineSearch<float, Paraboloid<float>>> optimizer;
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
        GradientDescent<float, Rosenbrock<float>> optimizer;
        optimizer.setMaxIterations(3000);
        Vector xval(2);
        xval << -0.5, 0.5;
        Vector xvalExp(2);
        xvalExp << 1, 1;

        auto result = optimizer.minimize(xval);
        REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
    }
}
