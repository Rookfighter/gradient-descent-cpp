/*
 * gradient_descent_cpp.h
 *
 * Author: Fabian Meyer
 * Created On: 12 Jul 2019
 */

#ifndef GRADIENT_DESCENT_CPP_H_
#define GRADIENT_DESCENT_CPP_H_

#include <Eigen/Geometry>
#include <limits>

namespace gdc
{
    typedef long int Index;

    template<typename Scalar,
        typename Objective>
    struct ForwardDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Objective objective;
        Scalar eps;

        ForwardDifferences()
            : objective(),
            eps(std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        ForwardDifferences(const Scalar eps)
            : objective(), eps(eps)
        {

        }

        void operator()(const Vector &state,
            const Scalar fval,
            Vector &gradient) const
        {
            Vector tmp;
            Vector nstate = state;

            gradient.resize(state.size());
            for(Index i = 0; i < state.size(); ++i)
            {
                nstate(i) += eps;
                Scalar fvalNew = objective(nstate, tmp);
                nstate(i) = state(i);

                gradient(i) = (fvalNew - fval) / eps;
            }
        }
    };

    template<typename Scalar,
        typename Objective>
    struct BackwardDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Objective objective;
        Scalar eps;

        BackwardDifferences()
            : objective(),
            eps(std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        BackwardDifferences(const Scalar eps)
            : objective(), eps(eps)
        {

        }

        void operator()(const Vector &state,
            const Scalar fval,
            Vector &gradient) const
        {
            Vector nstate = state;
            Vector tmp;

            gradient.resize(state.size());
            for(Index i = 0; i < state.size(); ++i)
            {
                nstate(i) -= eps;
                Scalar fvalNew = objective(nstate, tmp);
                nstate(i) = state(i);

                gradient(i) = (fval - fvalNew) / eps;
            }
        }
    };

    template<typename Scalar,
        typename Objective>
    struct CentralDifferences
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        Objective objective;
        Scalar eps;

        CentralDifferences()
            : objective(), eps(std::sqrt(std::numeric_limits<Scalar>::epsilon()))
        {

        }

        CentralDifferences(const Scalar eps)
            : objective(), eps(eps)
        {

        }

        void operator()(const Vector &state,
            const Scalar,
            Vector &gradient) const
        {
            Vector nstate = state;
            Vector tmp;

            gradient.resize(state.size());
            for(Index i = 0; i < state.size(); ++i)
            {
                nstate(i) = state(i) + eps / 2;
                Scalar fvalA = objective(nstate, tmp);
                nstate(i) = state(i) - eps / 2;
                Scalar fvalB = objective(nstate, tmp);
                nstate(i) = state(i);

                gradient(i) = (fvalA - fvalB) / eps;
            }
        }
    };

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
