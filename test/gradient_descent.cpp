/*
 * gradient_descent.cpp
 *
 * Author: Fabian Meyer
 * Created On: 12 Jul 2019
 */

#include "assert/eigen_require.h"
#include <gdc/gradient_descent.h>

using namespace gdc;

template<typename Scalar>
struct Paraboloid
{
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    Scalar operator()(const Vector &state, Vector &gradient) const
    {
        gradient.resize(2);
        gradient(0) = 2 * state(0);
        gradient(1) = 2 * state(1);

        return state(0) * state(0) + state(1) * state(1);
    }
};

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> Vector;

TEST_CASE("gradient_descent")
{
    const float eps = 1e-6;
    GradientDescent<float, Paraboloid<float>> optimizer;

    SECTION("optimize paraboloid")
    {
        Vector state(2);
        state << 2, 2;
        Vector stateExp(2);
        stateExp << 0, 0;

        auto result = optimizer.minimize(state);
        REQUIRE_MATRIX_APPROX(stateExp, result.state, eps);
    }
}
