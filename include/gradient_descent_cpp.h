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
#include <iostream>
#include <iomanip>

namespace gdc
{
    typedef long int Index;

    /** Functor to compute forward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + eps) - f(x)) / eps
      *
      * The computation requires len(x) evaluations of the objective.
      */
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

        void operator()(const Vector &xval,
            const Scalar fval,
            Vector &gradient) const
        {
            Vector tmp;
            Vector nxval = xval;

            gradient.resize(xval.size());
            for(Index i = 0; i < xval.size(); ++i)
            {
                nxval(i) += eps;
                Scalar fvalNew = objective(nxval, tmp);
                nxval(i) = xval(i);

                gradient(i) = (fvalNew - fval) / eps;
            }
        }
    };

    /** Functor to compute backward differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x) - f(x - eps)) / eps
      *
      * The computation requires len(x) evaluations of the objective.
      */
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

        void operator()(const Vector &xval,
            const Scalar fval,
            Vector &gradient) const
        {
            Vector nxval = xval;
            Vector tmp;

            gradient.resize(xval.size());
            for(Index i = 0; i < xval.size(); ++i)
            {
                nxval(i) -= eps;
                Scalar fvalNew = objective(nxval, tmp);
                nxval(i) = xval(i);

                gradient(i) = (fval - fvalNew) / eps;
            }
        }
    };

    /** Functor to compute central differences.
      * Computes the gradient of the objective f(x) as follows:
      *
      * grad(x) = (f(x + 0.5 eps) - f(x - 0.5 eps)) / eps
      *
      * The computation requires 2 * len(x) evaluations of the objective.
      */
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

        void operator()(const Vector &xval,
            const Scalar,
            Vector &gradient) const
        {
            Vector nxval = xval;
            Vector tmp;

            gradient.resize(xval.size());
            for(Index i = 0; i < xval.size(); ++i)
            {
                nxval(i) = xval(i) + eps / 2;
                Scalar fvalA = objective(nxval, tmp);
                nxval(i) = xval(i) - eps / 2;
                Scalar fvalB = objective(nxval, tmp);
                nxval(i) = xval(i);

                gradient(i) = (fvalA - fvalB) / eps;
            }
        }
    };

    /** Dummy callback functor, which does nothing. */
    template<typename Scalar>
    struct NoCallback
    {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        bool operator()(const Index,
            const Vector &,
            const Scalar,
            const Vector &) const
        {
            return true;
        }
    };

    template<typename Scalar,
        typename Objective,
        typename Callback=NoCallback<Scalar>,
        typename FiniteDifferences=CentralDifferences<Scalar, Objective> >
    class GradientDescent
    {
    public:
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

        struct Result
        {
            Index iterations;
            bool converged;
            Scalar fval;
            Vector xval;
        };
    protected:
        Index maxIt_;
        Scalar minGradientLen_;
        Scalar learningRate_;
        Scalar momentumRate_;
        bool verbose_;
        Objective objective_;
        Callback callback_;
        FiniteDifferences finiteDifferences_;

        Scalar evaluateObjective(const Vector &xval, Vector &gradient) const
        {
            gradient.resize(0);
            Scalar fval = objective_(xval, gradient);
            if(gradient.size() == 0)
                finiteDifferences_(xval, fval, gradient);
            return fval;
        }

        std::string vector2str(const Vector &vec) const
        {
            std::stringstream ss1;
            ss1 << std::fixed << std::showpoint << std::setprecision(6);
            std::stringstream ss2;
            ss2 << '[';
            for(Index i = 0; i < vec.size(); ++i)
            {
                ss1 << vec(i);
                ss2 << std::setfill(' ') << std::setw(10) << ss1.str();
                if(i != vec.size() - 1)
                    ss2 << ' ';
                ss1.str("");
            }
            ss2 << ']';

            return ss2.str();
        }

    public:

        GradientDescent()
            : maxIt_(0), minGradientLen_(1e-6), learningRate_(0.7),
            momentumRate_(0.9), verbose_(false), objective_(), callback_(),
            finiteDifferences_()
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

        void setCallback(const Callback &callback)
        {
            callback_ = callback;
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

        void setVerbose(const bool verbose)
        {
            verbose_ = verbose;
        }

        virtual Result minimize(const Vector &initialGuess) const
        {
            Vector gradient;
            Vector xval = initialGuess;

            Scalar fval = evaluateObjective(xval, gradient);
            Scalar gradientLen = gradient.norm();
            Vector step = learningRate_ * gradient;

            Index iterations = 0;
            while((maxIt_ <= 0 || iterations < maxIt_) && gradientLen >= minGradientLen_)
            {
                xval -= step;
                fval = evaluateObjective(xval, gradient);
                gradientLen = gradient.norm();
                step = momentumRate_ * step + learningRate_ * gradient;

                if(verbose_)
                {
                    std::cout << "it=" << std::setfill('0')
                        << std::setw(4) << iterations
                        << std::fixed << std::showpoint << std::setprecision(6)
                        << "    gradlen=" << gradientLen
                        << "    fval=" << fval
                        << "    xval=" << vector2str(xval)
                        << "    gradient=" << vector2str(gradient)
                        << "    step=" << vector2str(step)
                        << std::endl;
                }

                if(!callback_(iterations, xval, fval, gradient))
                    break;

                ++iterations;
            }

            Result result;
            result.xval = xval;
            result.fval = fval;
            result.iterations = iterations;
            result.converged = gradientLen < minGradientLen_;

            return result;
        }
    };
}

#endif
