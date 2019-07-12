/*
 * gradient_descent.h
 *
 * Author: Fabian Meyer
 * Created On: 12 Jul 2019
 */

#ifndef GDC_GRADIENT_DESCENT_H_
#define GDC_GRADIENT_DESCENT_H_

#include "gdc/finite_differences.h"

namespace gdc
{
    template<typename Scalar,
        typename Objective,
        typename FiniteDifferences=CentralDifferences<Scalar, Objective> >
    class GradientDescent
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        struct Result
        {
            Index iterations;
            bool converged;
            Scalar loss;
            Vector state;
        };
    protected:
        Index maxIt_;
        Scalar minGradientLen_;
        Scalar learningRate_;
        Scalar momentumRate_;
        Objective objective_;
        FiniteDifferences finiteDifferences_;

        Scalar evaluateObjective(const Vector &state, Vector &gradient) const
        {
            Scalar loss = objective_(state, gradient);
            if(gradient.size() == 0)
                finiteDifferences_(state, loss, gradient);
            return loss;
        }

        bool isMaxIteration(const Index iterations) const
        {
            return maxIt_ > 0 && iterations >= maxIt_;
        }

        bool hasConverged(const Vector &gradient)
        {
            return gradient.norm() < minGradientLen_;
        }

    public:

        GradientDescent()
            : maxIt_(0), minGradientLen_(1e-6), learningRate_(0.7),
            momentumRate_(0.9), objective_(), finiteDifferences_()
        {

        }

        virtual ~GradientDescent()
        {

        }

        void setMaximumIterations(const Index iterations)
        {
            maxIt_ = iterations;
        }

        void setObjective(const Objective &objective)
        {
            objective_ = objective;
            finiteDifferences_.objective = objective;
        }

        void setNumericalEpsilon(const Scalar eps)
        {
            finiteDifferences_.eps = eps;
        }

        void setMinimumGradientLength(const Scalar gradientLen)
        {
            minGradientLen_ = minGradientLen_;
        }

        void setLearningRate(const Scalar learningRate)
        {
            learningRate_ = learningRate;
        }

        void setMomentumRate(const Scalar momentumRate)
        {
            momentumRate_ = momentumRate;
        }

        virtual Result minimize(const Vector &state) const
        {
            Vector gradient(state.size());
            Vector step(state.size());
            Vector lastStep(state.size());
            Vector nstate = state;

            Scalar loss = evaluateObjective(state, gradient);
            Scalar gradientLen = gradient.norm();
            Index iterations = 0;
            lastStep.setZero();

            while((maxIt_ <= 0 || iterations < maxIt_) && gradientLen >= minGradientLen_)
            {
                step = momentumRate_ * lastStep + learningRate_ * gradient;
                nstate -= step;
                lastStep = step;
            }

            Result result;
            result.state = nstate;
            result.loss = loss;
            result.iterations = iterations;
            result.converged = gradientLen < minGradientLen_;

            return result;
        }
    };
}

#endif
