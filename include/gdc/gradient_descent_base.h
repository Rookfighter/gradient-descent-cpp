/*
 * gradient_descent_base.h
 *
 * Author: Fabian Meyer
 * Created On: 12 Jul 2019
 */

#ifndef GDC_GRADIENT_DESCENT_BASE_H_
#define GDC_GRADIENT_DESCENT_BASE_H_

namespace gdc
{
    typedef long int Index;

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

    template<typename Scalar,
        typename Objective>
    class GradientDescentBase
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    private:
        Index maxIt_;
        Scalar minGradientLen_;
        Objective objective_;

    public:
        GradientDescentBase()
            : maxIt_(0), minGradientLen_(1e-6), objective_()
        {

        }

        virtual ~GradientDescentBase()
        {

        }

        void setMaximumIterations(const Index iterations)
        {
            maxIt_ = iterations;
        }

        void setObjective(const Objective &objective)
        {
            objective_ = objective;
        }

        void setMinimumGradientLength(const Scalar gradientLen)
        {
            minGradientLen_ = minGradientLen_;
        }

        virtual void calculateStep(const Vector &state,
            const Vector &gradient,
            Vector &step) = 0;

        void minimize(Vector &state)
        {
            Index iterations = 0;

            Vector gradient(state.size());
            Vector step;

            while(maxIt_ <= 0 || iterations < maxIt_)
            {
                gradient.resize(0);
                Scalar value = objective_(state, gradient);

                if(gradient.size() == 0)
                {
                    // use finite differences
                }

                if(gradient.norm() < minGradientLen_)
                    break;

                calculateStep(state, gradient, step);

                state += step;

                ++iterations;
            }
        }
    };
}

#endif
