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
    Scalar operator()(const Vector &state, Vector &gradient)
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
        Vector xval(2);
        xval << 2, 2;
        Vector xvalExp(2);
        xvalExp << 0, 0;

        auto result = optimizer.minimize(xval);
        REQUIRE_MATRIX_APPROX(xvalExp, result.xval, eps);
    }
}
